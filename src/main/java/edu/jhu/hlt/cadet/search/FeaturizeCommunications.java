package edu.jhu.hlt.cadet.search;

import static java.lang.Integer.signum;
import static java.lang.System.exit;
import static java.lang.System.out;

import java.io.File;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.IOFileFilter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

import edu.jhu.hlt.concrete.Communication;
import edu.jhu.hlt.concrete.EntityMention;
import edu.jhu.hlt.concrete.Sentence;
import edu.jhu.hlt.concrete.TokenRefSequence;
import edu.jhu.hlt.concrete.Tokenization;
import edu.jhu.hlt.concrete.UUID;
import edu.jhu.hlt.concrete.serialization.CompactCommunicationSerializer;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;

/**
 * Created by rastogi on 6/10/17.
 */
enum FTag {
  ANCHOR_, MEMBER_;
}

public class FeaturizeCommunications implements Serializable {
  private final static Logger logger = LoggerFactory.getLogger(FeaturizeCommunications.class);

  private static Map<String, Integer> name2Id = new HashMap<>();
  private static Map<String, Integer> featHash = new HashMap<>();
  private static Map<String, Integer> featCount = new HashMap<>();


  public static Iterable<Map.Entry<Path, Communication>> getCommIter(String inDir) {
    IOFileFilter ioff = new IOFileFilter() {
      @Override
      public boolean accept(File file) {
        return true;
      }

      @Override
      public boolean accept(File file, String string) {
        return true;
      }
    };

    Iterator<File> fileIterator = FileUtils.iterateFiles(new File(inDir), ioff, null);
    CompactCommunicationSerializer ccs = new CompactCommunicationSerializer();
    return new Iterable<Map.Entry<Path, Communication>>() {
      @Override
      public Iterator<Map.Entry<Path, Communication>> iterator() {
        return new Iterator<Map.Entry<Path, Communication>>() {
          @Override
          public boolean hasNext() {
            return fileIterator.hasNext();
          }

          @Override
          public Map.Entry<Path, Communication> next() {
            try {
              Path path = fileIterator.next().toPath();
              return new java.util.AbstractMap.SimpleEntry<>(path, ccs.fromBytes(Files.readAllBytes(path)));
            } catch (Exception e) {
              e.printStackTrace();
              exit(1);
            }
            return null;
          }
        };
      }
    };
  }

  static void updateFeatState(String feature) {
    // featHash.putIfAbsent(feature, featHash.size());
    featCount.put(feature, featCount.getOrDefault(feature, 0) + 1);
  }

  static Map<UUID, List<String>> mapUUIDToTokens(Communication comm) {
    Map<UUID, List<String>> uuid2Tokens = new HashMap<>();
    for (Sentence sentence : comm.getSectionList().get(0).getSentenceList()) {
      Tokenization tokenization = sentence.getTokenization();
      List<String> tokens = new ArrayList<>();
      tokenization.getTokenList().getTokenList().forEach(x -> tokens.add(x.getText().toLowerCase()));
      uuid2Tokens.put(tokenization.getUuid(), tokens);
    }
    return uuid2Tokens;
  }

  static void initializeNameAndFeatureHash(String inDir) {
    int nameIdx = -1;
    for (Map.Entry<Path, Communication> pathComm : getCommIter(inDir)) {
      nameIdx++;
      Path path = pathComm.getKey();
      Communication comm = pathComm.getValue();
      name2Id.put(path.getFileName().toString().replace(".comm", ""), nameIdx);

      Map<UUID, List<String>> uuid2Tokens = mapUUIDToTokens(comm);

      List<EntityMention> entityMentionList = comm.getEntityMentionSetList().get(0).getMentionList();
      String feature;
      for (EntityMention em : entityMentionList) {
        TokenRefSequence trs = em.getTokens();
        List<Integer> tokenIdi = trs.getTokenIndexList();
        Integer anchorTokenIndex = trs.getAnchorTokenIndex();
        List<String> sentence = uuid2Tokens.get(trs.getTokenizationId());

        feature = FTag.ANCHOR_.toString() + sentence.get(anchorTokenIndex);
        updateFeatState(feature);
        for (Integer idx : tokenIdi) {
          feature = FTag.MEMBER_.toString() + sentence.get(idx);
          updateFeatState(feature);
        }
      }
    }
    List<String> features = new ArrayList<>();
    featCount.keySet().forEach(x -> features.add(x));
    features.sort(new Comparator<String>() {
      @Override
      public int compare(String o1, String o2) {
        return signum(featCount.get(o1) - featCount.get(o2));
      }
    });
    for (int index = 0; index < features.size(); index++) {
      featHash.put(features.get(index), index);
    }
    logger.info("featCount.size = {}", featCount.size());
    logger.info("featHash.size = {}", featHash.size());
    logger.info("name2Id.size = {}", name2Id.size());
  }

  static class Opts {
    @Parameter(names = { "--input-directory", "-i" }, description = "The path to the unzipped folder with concrete entities.")
    String inputDir;

    @Parameter(names = { "--output-filename", "-o" }, description = "The path to/name of the output file.")
    String outFileName = "onlyTokens.serial";

    @Parameter(names = { "--help", "-h" }, help = true, description = "Print the usage information and exit.")
    boolean help;
  }

  public static void main(String[] args) {
    Opts opts = new Opts();
    JCommander jc = null;
    try {
      jc = new JCommander(opts, args);
    } catch (ParameterException e) {
      System.err.println("Error parsing parameter: " + e.getMessage());
      exit(-1);
    }

    jc.setProgramName("FeaturizeCommunications");
    if (opts.help) {
      jc.usage();
      return;
    }

    try {
      initializeNameAndFeatureHash(opts.inputDir);
      FlexCompRowMatrix feat = new FlexCompRowMatrix(name2Id.size(), featHash.size());
      int docIndex = 0;
      for (Map.Entry<Path, Communication> pathComm : getCommIter(opts.inputDir)) {
        docIndex++;
        if (docIndex % 100 == 0)
          out.print('.');
        if (docIndex % 10000 == 0)
          out.println();
        Path path = pathComm.getKey();
        Integer rowIndex = name2Id.get(path.getFileName().toString().replace(".comm", ""));
        if (rowIndex == null) {
          out.println(path.toString());
          exit(1);
        }
        Communication comm = pathComm.getValue();
        Map<UUID, List<String>> uuid2Tokens = mapUUIDToTokens(comm);
        List<EntityMention> entityMentionList = comm.getEntityMentionSetList().get(0).getMentionList();
        String feature;
        for (EntityMention em : entityMentionList) {
          TokenRefSequence trs = em.getTokens();
          List<Integer> tokenIdi = trs.getTokenIndexList();
          Integer anchorTokenIndex = trs.getAnchorTokenIndex();
          List<String> sentence = uuid2Tokens.get(trs.getTokenizationId());

          feature = FTag.ANCHOR_.toString() + sentence.get(anchorTokenIndex);
          int colIndex = featHash.get(feature);
          MatrixUtils.increment(feat, rowIndex, colIndex);
          for (Integer idx : tokenIdi) {
            feature = FTag.MEMBER_.toString() + sentence.get(idx);
            colIndex = featHash.get(feature);
            MatrixUtils.increment(feat, rowIndex, colIndex);
          }
        }
      }

      CompRowMatrix binarizedFeat = new CompRowMatrix(MatrixUtils.binarize(feat, 2));
      EntityFeatureCorpus.validateFeatures(binarizedFeat);
      EntityFeatureCorpus efc = new EntityFeatureCorpus(binarizedFeat, name2Id, featHash);

      logger.info("Writing output.");
      efc.toFile(opts.outFileName);
      logger.info("Done.");
    } catch (Exception e) {
      logger.error("Caught exception running program", e);
      exit(1);
    }
  }
}
