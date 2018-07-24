package edu.jhu.hlt.cadet.search;

import static edu.jhu.hlt.cadet.search.EntityFeatureCorpus.validateFeatures;
import static edu.jhu.hlt.cadet.search.MatrixUtils.*;
import static java.lang.Math.signum;

import java.util.*;

import no.uib.cipr.matrix.VectorEntry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.ImmutableList;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.CompRowMatrix;

/**
 * Created by rastogi on 6/10/17.
 */
public class BinaryBayesianSets implements BayesianSets {
  private static final Logger logger = LoggerFactory.getLogger(BinaryBayesianSets.class);

  private final CompRowMatrix feat;
  private final Map<String, Integer> name2Id;
  private final List<String> names;
  private final DenseVector alpha;
  private final DenseVector beta;
  private final DenseVector logAlpha;
  private final DenseVector logBeta;
  private final int nItems;
  private final int nFeatures;
  private final Map<Integer, Integer> new2Old = new HashMap<>();
  BinaryBayesianSets(CompRowMatrix feat, Map<String, Integer> name2Id) {
    this(feat, name2Id, 2.0);
  }

  BinaryBayesianSets(CompRowMatrix feat, Map<String, Integer> name2Id, double c) {
    Map<Integer, Integer> old2New = old2New(feat);
    old2New.forEach((key, val) -> new2Old.put(val, key));
    feat = pruneZeroCols(feat, old2New);
    validateFeatures(feat);
    this.feat = feat;
    this.nItems = feat.numRows();
    this.nFeatures = feat.numColumns();
    this.name2Id = name2Id;
    this.alpha = sumRows(feat).scale(c / nItems);
    this.beta = sub(constant(nFeatures, c), alpha);
    this.logAlpha = log(alpha.copy());
    this.logBeta = log(beta.copy());
    names = new ArrayList<>();
    name2Id.keySet().forEach(s -> names.add(s));
    logger.debug("names.size = {}", names.size());
  }

  Pair<Vector, Vector> getScores(Collection<String> querySet) {
    List<Integer> idi = new ArrayList<>();
    for (String q : querySet)
      if (name2Id.containsKey(q))
        idi.add(name2Id.get(q));
      else
        logger.info("Name2ID map does not contain entry: {}", q);
    if (idi.isEmpty())
      return null;

    idi.sort(Integer::compareTo);
    Vector sumQueryFeat = sumRows(feat, idi);
    DenseVector logBetaTilde = log(
        sub((DenseVector) constant(nFeatures, (double) querySet.size()).add(this.beta), sumQueryFeat));
    // sumQueryFeat is now corrupt. It contains alphatilde.
    DenseVector logAlphaTilde = log(new DenseVector(sumQueryFeat.add(this.alpha)));
    Vector q = sub(logAlphaTilde.add(logBeta), logBetaTilde.add(logAlpha));
    Vector scores = zero(nItems);
    feat.mult(q, scores);
    return new Pair<Vector, Vector>(scores, q);
  }

  public BayesianSetsQueryResult query(Collection<String> querySet) {
    ImmutableList.Builder<StringScoreTuple> b = ImmutableList.builder();
    Pair<Vector, Vector> scores_q = getScores(querySet);
    if (scores_q == null)
      return new BayesianSetsQueryResult(new ArrayList<>(), new HashMap<Integer, Double>());
    Vector scores = scores_q.getKey();
    Vector q = scores_q.getValue();
    Vector unpacked = scores; // .get();
    names.sort((o1, o2) -> {
      Double s2 = unpacked.get(name2Id.get(o2));
      Double s1 = unpacked.get(name2Id.get(o1));
      return (int) signum(s2 - s1);
    });

    for (String name : names) {
      StringScoreTuple sst = new StringScoreTuple.Builder().setString(name).setScore(unpacked.get(name2Id.get(name))).build();
      // out.println(sst);
      b.add(sst);
    }
    ImmutableList<StringScoreTuple> eScores = b.build();
    Map<Integer, Double> lm = new HashMap<>();
    for(VectorEntry e: q){
      lm.put(new2Old.get(e.index()), e.get());
    }
    return new BayesianSetsQueryResult(eScores, lm);
  }
}
