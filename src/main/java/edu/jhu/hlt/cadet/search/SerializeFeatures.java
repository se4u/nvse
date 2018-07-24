package edu.jhu.hlt.cadet.search;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Stream;

import static java.lang.System.exit;
import static java.lang.System.out;

/**
 * Created by rastogi on 9/8/17.
 */


public class SerializeFeatures {
  private final static Logger logger = LoggerFactory.getLogger(FeaturizeCommunications.class);

  public static void main(String[] args) {
    String flag = args[0];
    String outputFile = args[1];
    String featValFn = args[2];
    String entityMapFn = args[3];
    String featMapFn = args[4];
    Double thresholdFactor = 0.5; // Good values are between 0 and 1; 0, 0.5, 1
    if(args.length == 6)
      thresholdFactor = Double.valueOf(args[5]);
    Map<String, Integer> name2Id = new HashMap<>();
    try (Stream<String> lines = Files.lines(Paths.get(entityMapFn))) {
      int[] i = {0};
      lines.forEachOrdered(x -> {
        name2Id.put(x.trim().split("\t")[0], i[0]);
        i[0] = i[0] + 1;
      });
    } catch (Exception e) {
      exit(1);
    }

    Map<String, Integer> featHash = new HashMap<>();
    try (Stream<String> lines = Files.lines(Paths.get(featMapFn))) {
      int[] i = {0};
      lines.forEachOrdered(x -> {
        featHash.put(x.trim().split(" ")[0], i[0]);
        i[0] = i[0] + 1;
      });
    } catch (Exception e) {
      exit(1);
    }

    FlexCompRowMatrix feat = new FlexCompRowMatrix(name2Id.size(), featHash.size());
    try (Stream<String> lines = Files.lines(Paths.get(featValFn))) {
      int[] i = {0};
      lines.forEachOrdered(x -> {
        // if(i[0] % 100 == 0)
        //  out.println(i[0] / 778.45);
        String[] tokens = x.split(" ");
        for(int j = 1; j < tokens.length; j++){
          String[] feat_count = tokens[j].split(":");
          Integer feature = Integer.valueOf(feat_count[0]);
          Integer count = Integer.valueOf(feat_count[1]);
          feat.set(i[0], feature, count);
        }
        i[0] = i[0] + 1;
      });
    } catch (Exception e) {
      exit(1);
    }

    CompRowMatrix features = null;
    EntityFeatureCorpus efc = null;
    if(flag.equals("binary")) {
      features = new CompRowMatrix(MatrixUtils.binarize(feat, thresholdFactor));
    } else if (flag.equals("gaussian"))  {
      features = new CompRowMatrix(MatrixUtils.normalize(MatrixUtils.sqrt(feat)));
    }
    logger.info("Zero columns in features originally: " + String.valueOf(MatrixUtils.zeroColumns(features).size()));
    // Prune out the bad columns.
    Map<Integer, Integer> old2new = MatrixUtils.old2New(features);
    features = MatrixUtils.pruneZeroCols(features, old2new);
    Map<String, Integer> newFeatHash = new HashMap<>();
    for (Map.Entry<String, Integer> entry : featHash.entrySet())
      if(old2new.containsKey(entry.getValue()))
        newFeatHash.put(entry.getKey(), old2new.get(entry.getValue()));
    // Check that the new matrix does not contain bad features.
    if(MatrixUtils.zeroColumns(features).size() > 0) {
      logger.error("MatrixUtils.zeroColumns(features).size() = " + String.valueOf(MatrixUtils.zeroColumns(features).size()));
      exit(1);
    }
    efc = new EntityFeatureCorpus(features, name2Id, newFeatHash);
    try {
      logger.info("Writing output to " + outputFile);
      efc.toFile(outputFile);
    } catch (Exception e) {
      logger.error("Caught exception running program", e);
      exit(1);
    }
  }
}
