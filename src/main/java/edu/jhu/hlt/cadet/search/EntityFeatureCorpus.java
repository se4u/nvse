package edu.jhu.hlt.cadet.search;

import static java.lang.System.exit;
import static java.lang.System.out;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by rastogi on 6/11/17.
 */
public class EntityFeatureCorpus implements Serializable {
  private final static Logger logger = LoggerFactory.getLogger(FeaturizeCommunications.class);
  public final CompRowMatrix feat;
  public final Map<String, Integer> name2id;
  public final Map<String, Integer> featHash;

  public EntityFeatureCorpus(CompRowMatrix feat, Map<String, Integer> name2id, Map<String, Integer> featHash) {
    this.feat = feat;
    this.name2id = name2id;
    this.featHash = featHash;
  }

  public void toFile(String fn) throws FileNotFoundException, IOException {
    try(FileOutputStream fs = new FileOutputStream(fn);
        BufferedOutputStream bout = new BufferedOutputStream(fs);
        ObjectOutputStream oos = new ObjectOutputStream(bout);) {
      logger.info("Validation Started");
      validateFeatures(feat);
      logger.info("Serialization Started");
      MatrixUtils.serializeCSR(oos, feat);
      logger.info("Name2Id Started");
      oos.writeObject(name2id);
      logger.info("FeatHash Started");
      oos.writeObject(featHash);
      oos.close();
    }
  }

  static void validateFeatures(CompRowMatrix feat) {
    double matSum = MatrixUtils.sumAll(feat);

    boolean allLtOne = true;
    for(MatrixEntry e: feat)
      allLtOne = allLtOne & (e.get() <= 1);

    logger.info("Math.abs(matSum): " + String.valueOf(Math.abs(matSum)));
    logger.info("All elem less than one: " + String.valueOf(allLtOne));
    if(!allLtOne & Math.abs(matSum) > 1e-7) {
      logger.error("!allLtOne & Math.abs(matSum) > 1e-7");
      exit(1);
    }

    // Check that all the columns have atleast one feature value.
    int zc = MatrixUtils.zeroColumns(feat).size();
    if(zc != 0) {
      logger.error("Zero columns: " + String.valueOf(zc) + " Out of: " + String.valueOf(feat.numColumns()));
      exit(1);
    }
  }

  public static EntityFeatureCorpus fromFile(String fn) throws FileNotFoundException, IOException, ClassNotFoundException {
    try (FileInputStream fs = new FileInputStream(new File(fn));
        BufferedInputStream bin = new BufferedInputStream(fs);
        ObjectInputStream oos = new ObjectInputStream(bin);) {

      CompRowMatrix feat = MatrixUtils.deserializeCSR(oos);
      validateFeatures(feat);
      Map<String, Integer> name2id = (Map<String, Integer>) oos.readObject();
      logger.info("name2Id.size = " + String.valueOf(name2id.size()));
      Map<String, Integer> featHash = (Map<String, Integer>) oos.readObject();
      logger.info("featHash.size = " + String.valueOf(featHash.size()));
      return new EntityFeatureCorpus(feat, name2id, featHash);
    }
  }
}
