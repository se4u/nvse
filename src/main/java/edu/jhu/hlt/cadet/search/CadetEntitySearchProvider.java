package edu.jhu.hlt.cadet.search;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import edu.jhu.hlt.concrete.AnnotationMetadata;
import edu.jhu.hlt.concrete.search.SearchCapability;
import edu.jhu.hlt.concrete.search.SearchQuery;
import edu.jhu.hlt.concrete.search.SearchResult;
import edu.jhu.hlt.concrete.search.SearchResultItem;
import edu.jhu.hlt.concrete.search.SearchService;
import edu.jhu.hlt.concrete.search.SearchType;
import edu.jhu.hlt.concrete.services.ServiceInfo;
import edu.jhu.hlt.concrete.services.ServicesException;
import edu.jhu.hlt.concrete.uuid.AnalyticUUIDGeneratorFactory;

import static edu.jhu.hlt.cadet.search.MatrixUtils.retrieveObjectWithHighestScore;
import static java.lang.Math.min;
import static java.lang.System.exit;
import static java.lang.System.out;

public class CadetEntitySearchProvider implements SearchService.Iface, AutoCloseable {
  private static Logger logger = LoggerFactory.getLogger(CadetEntitySearchProvider.class);
  private final AnalyticUUIDGeneratorFactory.AnalyticUUIDGenerator uuidGen;
  private final String languageCode;
  private final Opts opts;
  private final Integer kQuery;
  private final Integer kRationale;
  private final Map<String, Integer> name2Id;
  private final Map<Integer, String> id2Feat = new HashMap<>();
  // private final Map<Integer, String>
  private List<EntitySentencesAndFeatures> corpus = new ArrayList<>();
  // private final EntityFeatureCorpus index;
  public BayesianSets bayesianSets;

  public CadetEntitySearchProvider(String languageCode, EntityFeatureCorpus index, Opts opts) {
    this.languageCode = languageCode;
    this.opts = opts;
    AnalyticUUIDGeneratorFactory f = new AnalyticUUIDGeneratorFactory();
    this.uuidGen = f.create();
    kRationale = opts.kRationale;
    kQuery = opts.kQuery;
    name2Id = index.name2id;
    index.featHash.forEach((key, val) -> id2Feat.put(val, key));
    try (Stream<String> lines = Files.lines(Paths.get(opts.sFile))){
      Integer cntr = -1;
      for(String line : lines.collect(Collectors.toList())){
        cntr ++;
        //if(cntr % 10000 == 0) out.println(cntr);
        String[] parts = line.split(" \\|\\|\\|");
        Integer entityId = index.name2id.get(parts[0].trim());
        List<String> sentences = new ArrayList<>();
        List<Integer[]> sentenceFeatures = new ArrayList<>();
        if(!entityId.equals(cntr)){
          out.println("Error");
          exit(1);
        }
        for(int i = 1;i < parts.length; i++){
          String[] tokenizedSentence = parts[i].split(" ");
          Integer nFeat = 0;
          for(int j = 0; j < tokenizedSentence.length; j++)
            if(index.featHash.containsKey(tokenizedSentence[j]))
              nFeat++;

          Integer[] features = new Integer[nFeat];
          nFeat = -1;
          for(int j = 0; j < tokenizedSentence.length; j++) {
            if (index.featHash.containsKey(tokenizedSentence[j])) {
              nFeat++;
              features[nFeat] = index.featHash.get(tokenizedSentence[j]);
            }
          }
          sentences.add(parts[i]);
          sentenceFeatures.add(features);
        }
        corpus.add(new EntitySentencesAndFeatures(sentences, sentenceFeatures));
      }
    } catch (IOException e) { e.printStackTrace(); exit(1); }

    switch (opts.algorithm){
      case BINARY:
        this.bayesianSets = new BinaryBayesianSets(index.feat, index.name2id); break;
//      case NEURAL:
//        this.bayesianSets = new NeuralBayesianSets(index.name2id, -1); break;
//      case GAUSSIAN:
//        this.bayesianSets = new GaussianBayesianSets(index.feat, index.name2id); break;
      default:
        logger.error("Bad configuration of algorithm: " + opts.algorithm);
        exit(1);
    }
  }

  @Override
  public SearchResult search(SearchQuery query) throws ServicesException, TException {
    final int maxResults = query.isSetK() ? query.getK() : Integer.MAX_VALUE;
    List<String> terms = query.getTerms();
    if (query.getRawQuery().trim().equals("")) {
      logger.info("Short circuiting an empty query");
      return createResultsContainer(query).setSearchResultItems(new ArrayList<SearchResultItem>());
    }
    logger.info("Search query: " + query.getRawQuery());
    try {
      BayesianSetsQueryResult queryResult = bayesianSets.query(terms);
      SearchResult results = createResultsContainer(query, queryResult.lm);
      List<StringScoreTuple> sortable = new ArrayList<>(queryResult.eScores);
      Collections.sort(sortable, StringScoreTuple.descendingScoreComparator());

      for (StringScoreTuple sst : sortable) {
        SearchResultItem result = new SearchResultItem();
        if(queryResult.lm != null) {
          EntitySentencesAndFeatures sentencesAndFeatures = corpus.get(name2Id.get(sst.getString()));
          List<Integer[]> sentenceFeatures = sentencesAndFeatures.sentenceFeatures;
          List<String> sentences = sentencesAndFeatures.sentences;
          Double[] scores = new Double[sentenceFeatures.size()];
          int sentenceIdx = 0;
          for (Integer[] x : sentenceFeatures) {
            double sentScore = 0;
            for (int i = 0; i < x.length; i++)
              sentScore += (queryResult.lm.get(x[i]) - sentScore)/(i+1);
            scores[sentenceIdx] = sentScore * Math.log(Math.min(x.length, 20));
            sentenceIdx++;
          }

          List<String> highScoreSentences = retrieveObjectWithHighestScore(sentences, scores, kQuery);
          result.setCommunicationId(sst.getString() + '\n' + String.join("\n", highScoreSentences));
        } else {
          result.setCommunicationId(sst.getString());
        }

        result.setSentenceId(null);
        result.setScore(sst.getScore());
        results.addToSearchResultItems(result);
        if (results.getSearchResultItemsSize() >= maxResults)
          break;
      }

      logger.info("Returning " + results.getSearchResultItemsSize() + " results");
      return results;
    } catch (Exception e) {
      throw new ServicesException(e.getMessage());
    }
  }

  private SearchResult createResultsContainer(SearchQuery query) {
    SearchResult results = new SearchResult();
    results.setUuid(uuidGen.next());
    results.setSearchQuery(query);
    AnnotationMetadata metadata = new AnnotationMetadata();
    metadata.setTool("Cadet Entity Recommender");
    results.setMetadata(metadata);
    return results;
  }

  private SearchResult createResultsContainer(SearchQuery query, Map<Integer, Double> lm) {
    SearchResult results = new SearchResult();
    results.setUuid(uuidGen.next());
    if(lm != null) {
      List<Integer> sortedFeatureIdx = lm.entrySet().stream()
        .sorted(Map.Entry.comparingByValue(Collections.reverseOrder()))
        .map(Map.Entry::getKey)
        .collect(Collectors.toList());

      for(int i = 0; i < min(kQuery, sortedFeatureIdx.size()); i++)
        query.addToLabels(id2Feat.get(sortedFeatureIdx.get(i)));
    }
    results.setSearchQuery(query);
    AnnotationMetadata metadata = new AnnotationMetadata();
    metadata.setTool("Cadet Entity Recommender");
    results.setMetadata(metadata);
    return results;
  }

  @Override
  public ServiceInfo about() throws TException {
    return new ServiceInfo("Cadet Entity Recommendation", "0.1.0");
  }

  @Override
  public boolean alive() throws TException {
    return true;
  }

  @Override
  public void close() throws IOException {
  }

  @Override
  public List<SearchCapability> getCapabilities() throws ServicesException, TException {
    List<SearchCapability> capabilities = new ArrayList<>();
    SearchCapability communicationsCapability = new SearchCapability();
    communicationsCapability.setLang(this.languageCode);
    communicationsCapability.setType(SearchType.ENTITIES);
    capabilities.add(communicationsCapability);
    return capabilities;
  }

  @Override
  public List<String> getCorpora() throws ServicesException, TException {
    return new ArrayList<String>();
  }
}
