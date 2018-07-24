package edu.jhu.hlt.cadet.search;

import static java.lang.System.exit;
import static java.lang.System.out;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;

import edu.jhu.hlt.concrete.AnnotationMetadata;
import edu.jhu.hlt.concrete.Communication;
import edu.jhu.hlt.concrete.EntityMention;
import edu.jhu.hlt.concrete.EntityMentionSet;
import edu.jhu.hlt.concrete.Section;
import edu.jhu.hlt.concrete.Sentence;
import edu.jhu.hlt.concrete.Token;
import edu.jhu.hlt.concrete.TokenRefSequence;
import edu.jhu.hlt.concrete.Tokenization;
import edu.jhu.hlt.concrete.UUID;
import edu.jhu.hlt.concrete.serialization.CompactCommunicationSerializer;
import edu.jhu.hlt.concrete.util.ConcreteException;
import edu.jhu.hlt.concrete.uuid.AnalyticUUIDGeneratorFactory;
import edu.jhu.hlt.concrete.zip.ConcreteZipIO;

/**
 * Created by rastogi on 6/8/17.
 */
class DefaultDict extends HashMap<String, List<JsonEntity>> {

  /**
   *
   */
  private static final long serialVersionUID = -1532813787262809293L;

  @Override
  public List<JsonEntity> get(Object key) {
    List<JsonEntity> returnValue = super.get(key);
    if (returnValue == null) {
      returnValue = new ArrayList<JsonEntity>();
      this.put((String) key, returnValue);
    }
    return returnValue;
  }
}

class Prov {
  public final String doc;
  public final String content;
  public final Long start;
  public final Long end;

  Prov(String doc, String content, Long start, Long end) {
    this.doc = doc;
    this.content = content;
    this.start = start;
    this.end = end;
  }

  @Override
  public String toString() {
    StringBuilder bldr = new StringBuilder();
    bldr.append("\n  doc:").append(doc).append("\n  content:").append(content).append("\n  start:").append(start)
        .append("\n  end:").append(end);
    return bldr.toString();
  }

  @Override
  public int hashCode() {
    return Objects.hash(doc, content, start, end);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    return (this.hashCode() == ((Prov) obj).hashCode());
  }
}

class MentionSentence {
  public final Sentence sent;
  public final TokenRefSequence trs;

  MentionSentence(Sentence sent, TokenRefSequence trs) {
    this.sent = sent;
    this.trs = trs;
  }
}

class JsonEntity {
  public final String cs;
  public final String kbid;
  public final List<String> relent;
  public final Set<Prov> provs;
  public List<MentionSentence> msList;
  private static final AnalyticUUIDGeneratorFactory.AnalyticUUIDGenerator uuidFactory = (new AnalyticUUIDGeneratorFactory())
      .create();

  JsonEntity(String cs, String kbid, List<String> relent, Set<Prov> provs) {
    this.cs = cs;
    this.kbid = kbid;
    this.relent = relent;
    this.provs = provs;
    this.msList = new ArrayList<>();
  }

  /**
   * @return Communication object with 1) sentences containing mentions of the
   *         entity in question (row) 2) exactly one EntityMentionSet pointing out
   *         the positions in the sentences referring to the entity in question
   */
  Communication toCommunication() {
    Section sec = new Section(uuidFactory.next(), "post");
    sec.setSentenceList(this.msList.stream().map(x -> x.sent).collect(Collectors.toList()));
    EntityMentionSet ems = new EntityMentionSet(uuidFactory.next(),
        new AnnotationMetadata("ADEPT Entity Linking", 1459294485, 1),
        this.msList.stream().map(x -> new EntityMention(uuidFactory.next(), x.trs)).collect(Collectors.toList()));

    Communication comm = new Communication(kbid, new UUID(kbid), "KBEntity",
        new AnnotationMetadata("BoltForumPostIngester 4.8.7-SNAPSHOT", 1459113967, 1));
    comm.setEntityMentionSetList(Arrays.asList(ems));
    comm.setSectionList(Arrays.asList(sec));
    return comm;
  }

  void toFile(String fn) throws ConcreteException, IOException {
    FileUtils.writeByteArrayToFile(new File(fn),
        (new CompactCommunicationSerializer()).toBytes(this.toCommunication()));
  }

  static List<String> getRelEnt(JSONObject jobj) {
    List<String> ret = new ArrayList<>();
    Iterator iter = null;
    try {
      iter = ((JSONArray) jobj.get("Related_ents")).iterator();
      while (iter.hasNext()) {
        ret.add(iter.next().toString());
      }
    } catch (NullPointerException e) {
    }
    return ret;
  }

  static Set<Prov> getProvSet(JSONObject jobj) {
    Set<Prov> ret = new HashSet<>();
    Iterator iter = ((JSONArray) jobj.get("ProvList")).iterator();
    while (iter.hasNext()) {
      JSONObject item = (JSONObject) iter.next();
      JSONObject item2 = (JSONObject) item.get("Right");
      String fn = ((String) item.get("Left"));
      Integer filenameOffset = fn.lastIndexOf("/") + 1;
      Prov prov = new Prov(fn.substring(filenameOffset).split("[.]")[0], (String) item2.get("Left"),
          (Long) item2.get("Middle"), (Long) item2.get("Right"));

      ret.add(prov);
    }
    return ret;
  }

  @Override
  public String toString() {
    StringBuilder bldr = new StringBuilder();
    bldr.append("\nCanonicalString:").append(cs).append("\nkbid:").append(kbid).append("\nRelEnt:").append(relent)
        .append("\nProvList:").append(provs);
    return bldr.toString();
  }

  public String toFlatFile() {
    StringBuilder bldr = new StringBuilder();
    bldr.append(kbid);
    for (Prov p : provs) {
      bldr.append(" ||| ").append(p.doc).append(" ").append(p.start).append(" ").append(p.end);
    }
    bldr.append("\n");
    return bldr.toString();
  }
}

/**
 * Create entity centric communication files into entityCommDir And create a
 * flat file representation into output_fn
 */
public class EntityWiseSerializeAdeptComm {
  private static final Logger log = LoggerFactory.getLogger(EntityWiseSerializeAdeptComm.class);

  private final String inputZipFilename;
  private final String outputFilename;
  private final String jsonFilename;

  public EntityWiseSerializeAdeptComm(String inputZipFilename, String outputFilename, String jsonFilename) {
    this.inputZipFilename = inputZipFilename;
    this.outputFilename = outputFilename;
    this.jsonFilename = jsonFilename;
  }

  void writeFlatFile(DefaultDict comm2ent, List<JsonEntity> jsonEntities)
      throws FileNotFoundException, IOException, ParseException {
    JSONParser jsonParser = new JSONParser();
    try (FileOutputStream fos = new FileOutputStream(this.outputFilename);
        OutputStreamWriter wrtr = new OutputStreamWriter(fos, StandardCharsets.UTF_8);) {
      try (FileInputStream fis = new FileInputStream(this.jsonFilename);
          InputStreamReader isr = new InputStreamReader(fis, StandardCharsets.UTF_8);) {
        Iterator jsonArray = ((JSONArray) jsonParser.parse(isr)).iterator();
        while (jsonArray.hasNext()) {
          JSONObject obj = (JSONObject) jsonArray.next();
          Set<Prov> provs = JsonEntity.getProvSet(obj);
          JsonEntity ent = new JsonEntity((String) obj.get("CanonicalString"), (String) obj.get("Kbid"),
              JsonEntity.getRelEnt(obj), provs);
          jsonEntities.add(ent);
          for (Prov prov : provs) {
            comm2ent.get(prov.doc).add(ent);
          }
          String str = ent.toFlatFile();
          wrtr.write(str);
        }
      }
    }
  }

  void populateJsonEntities(DefaultDict comm2ent) {
    Iterable<Communication> zio = ConcreteZipIO.readAsStream(inputZipFilename)::iterator;
    Integer idx = 0;
    for (Communication com : zio) {
      idx += 1;
      if (idx % 100 == 0) {
        out.print(".");
        out.flush();
      }
      if (idx % 10000 == 0) {
        out.println(".");
        out.flush();
      }
      String commId = com.getId();
      List<JsonEntity> jsonEntitiesList = comm2ent.get(commId);
      if (jsonEntitiesList == null) {
        out.println(commId);
        continue;
      }
      Map<UUID, Pair<TokenRefSequence, String>> uuid2trs = new HashMap<>();
      List<EntityMentionSet> emsList = com.getEntityMentionSetList();
      for (EntityMentionSet ems : emsList) {
        for (EntityMention em : ems.getMentionList()) {
          TokenRefSequence trs = em.getTokens();
          uuid2trs.put(trs.getTokenizationId(), Pair.of(trs, em.getText()));
        }
      }
      List<Section> sectionList = com.getSectionList();
      for (Section sec : sectionList) {
        List<Sentence> sentenceList = sec.getSentenceList();
        for (Sentence sent : sentenceList) {
          Tokenization tk = sent.getTokenization();
          UUID tkUUID = tk.getUuid();
          Pair<TokenRefSequence, String> trs_text = uuid2trs.get(tkUUID);
          if (trs_text == null) {
            continue;
          }
          TokenRefSequence trs = trs_text.getKey();
          List<Integer> tokenIndexList = trs.getTokenIndexList();
          Integer startTokenIdx = tokenIndexList.get(0);
          Integer endTokenIdx = tokenIndexList.get(tokenIndexList.size() - 1);
          List<Token> tokenList = tk.getTokenList().getTokenList();
          Integer startTokenOffset = tokenList.get(startTokenIdx).getTextSpan().getStart();
          Integer endTokenOffset = tokenList.get(endTokenIdx).getTextSpan().getEnding();
          String mentionText = trs_text.getValue();
          for (JsonEntity ent : jsonEntitiesList) {
            for (Prov prov : ent.provs) {
              if (prov.doc.equals(commId)) {
                Long start = prov.start;
                Long end = prov.end;
                if (start >= startTokenOffset && end <= endTokenOffset) {
                  ent.msList.add(new MentionSentence(sent, trs));
                }
              }
            }
          }
        }
      }
    }
  }

  static class Opts {
    @Parameter(names = { "--input-zip-path", "-ip" }, required = true, description = "The path to the input zip file.")
    String inputFile;

    @Parameter(names = { "--output-file", "-op" }, required = true, description = "Path to the output file.")
    String outputFile;

    @Parameter(names = { "--json-path", "-jp" }, required = true, description = "The path to the JSON file to read.")
    String jsonPath;

    @Parameter(names = { "--entity-communications-path",
        "-ecp" }, required = true, description = "The path to the communications holding the entities.")
    String entityCommsPath;

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
    jc.setProgramName("EntityWiseSerializeAdeptComm");
    if (opts.help) {
      jc.usage();
      return;
    }

    EntityWiseSerializeAdeptComm ews = new EntityWiseSerializeAdeptComm(opts.inputFile, opts.outputFile, opts.jsonPath);
    DefaultDict comm2ent = new DefaultDict();

    List<JsonEntity> jsonEntities = new ArrayList<>();
    try {
      ews.writeFlatFile(comm2ent, jsonEntities);
    } catch (IOException | ParseException e) {
      log.error("Caught exception writing file", e);
      System.exit(127);
    }

    out.println("Length of jsonEntities " + String.valueOf(jsonEntities.size()));
    ews.populateJsonEntities(comm2ent);
    try {
      out.println("Started Writing");
      int filesWritten = 0;
      for (JsonEntity jent : jsonEntities) {
        jent.toFile(opts.entityCommsPath + jent.kbid + ".comm");
        filesWritten++;
      }
      out.println(filesWritten);
    } catch (Exception e) {
      e.printStackTrace();
      exit(1);
    }
  }
}
