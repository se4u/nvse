package edu.jhu.hlt.cadet.search;

import com.google.common.collect.ImmutableList;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import org.apache.commons.math3.special.Gamma;
import org.apache.commons.math3.util.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.stream.IntStream;

import static edu.jhu.hlt.cadet.search.EntityFeatureCorpus.validateFeatures;
import static edu.jhu.hlt.cadet.search.MatrixUtils.zero;
import static java.lang.Math.signum;
/**
 * Created by rastogi on 9/9/17.
 */
public class GaussianBayesianSets  { // implements BayesianSets
  private static final Logger logger = LoggerFactory.getLogger(GaussianBayesianSets.class);

  private final CompRowMatrix feat;
  private final Map<String, Integer> name2Id;
  private final List<String> names;
  private final double rho;
  private final double lambda;
  private final double alpha;
  private final double beta;
  private final int nItems;
  private final int nFeatures;
  private final double[] xbar;
  private final double[] rhot;
  private final double[] betat;
  // private final FlexCompRowMatrix flexFeat;

  GaussianBayesianSets(CompRowMatrix feat, Map<String, Integer> name2Id) {
    this(feat, name2Id, 0, 2, 2, 1);
  }

  GaussianBayesianSets(CompRowMatrix feat, Map<String, Integer> name2Id,
                       double rho, double lambda, double alpha, double beta) {
    validateFeatures(feat);
    this.feat = feat;
    // flexFeat = new FlexCompRowMatrix(feat);
    nItems = feat.numRows();
    nFeatures = feat.numColumns();
    this.name2Id = name2Id;
    names = new ArrayList<>();
    name2Id.keySet().forEach(s -> names.add(s));
    logger.debug("names.size = {}", names.size());
    logger.info("Runtime.getRuntime().availableProcessors(): " + String.valueOf(Runtime.getRuntime().availableProcessors()));
    this.rho = rho;
    this.lambda = lambda;
    this.alpha = alpha;
    this.beta = beta;
    // Set up memory for the data structures.
    xbar = new double[this.nFeatures];
    rhot = new double[this.nFeatures];
    betat = new double[this.nFeatures];
  }

  double logFactor(double nu, double b){
    double nup1by2 = (nu+1)/2;
    double logFactor = Gamma.logGamma( nup1by2 ) -
      0.5 * (FastMath.log(FastMath.PI) + FastMath.log(nu) + FastMath.log(b)) -
      Gamma.logGamma(nu / 2);
    return logFactor;
  }

  double logT(double x, double nu, double a, double b, double logFactor){
    double xma = x-a;
    return (xma == 0)? logFactor : (logFactor - ((nu+1)/2 * (FastMath.log1p( xma * xma / b / nu ))));
  }

  double logT(double x, double nu, double a, double b){
    return logT( x, nu, a, b, logFactor(nu, b));
  }

  Optional<Vector> getScores(Collection<String> querySet) {
    List<Integer> idi = new ArrayList<>();
    for (String q : querySet) {
      if (name2Id.containsKey(q)) {
        idi.add(name2Id.get(q));
      } else {
        logger.error("Name2ID map does not contain entry: {}", q);
      }
    }
    if (idi.isEmpty()) return Optional.empty();
    idi.sort(Integer::compareTo);
    // Start preparing the features.
    int D = idi.size();
    double lambdat = lambda + D;
    double alphat = alpha + D/2.0;
    MatrixUtils.sumRows(xbar, feat, idi);
    for(int j = 0; j < xbar.length; j++) {
      xbar[j] /= D;
      rhot[j] = (rho * lambda + D * xbar[j]) / (lambdat);
    }
    MatrixUtils.sSE(betat, feat, idi, xbar);
    double hm_dlambda = (D*lambda)/lambdat/2;
    for(int j = 0; j < betat.length; j++){
      double delta = (xbar[j] - rhot[j]);
      betat[j] = beta + betat[j] / 2 + hm_dlambda * (delta * delta);
    }
    Vector scores = zero(nItems);
    double scaleFactorNum =  (1 + 1/lambdat) / alphat;
    double scaleFactorDen =  (1 + 1/lambda) / alpha;
    double[] logFactorArr = new double[nFeatures];
    for(int j = 0; j < nFeatures; j++)
      logFactorArr[j] = logFactor(2*alphat, betat[j] * scaleFactorNum);
    double logFactorDen = logFactor(2*alpha, beta * scaleFactorDen);
    for(int i = 0; i < nItems; i++){
      double logScore = 0;
      if(i % 1000 == 0)
        logger.info("Items Processed: "+String.valueOf(i));
      DoubleAdder a = new DoubleAdder();
      final int finalI = i;
      IntStream.range(0, nFeatures).parallel().forEach(j -> {
        double xj = feat.get(finalI, j);
        a.add((logT(xj, 2*alphat, rhot[j], betat[j] * scaleFactorNum, logFactorArr[j])
               - logT(xj, 2*alpha, rho, beta * scaleFactorDen, logFactorDen)));
      });
      logScore = a.doubleValue();
//      for(int j = 0; j < nFeatures; j++){
//        double xj = feat.get(i, j);
//        logScore += (logT(xj, 2*alphat, rhot[j], betat[j] * scaleFactorNum, logFactorArr[j])
//                     - logT(xj, 2*alpha, rho, beta * scaleFactorDen, logFactorDen));
//      }
      // TODO: One hack that can be done is to increment the logScore only for those elements that are non-zero.
      // TODO: Another possibility is simple multi-threading.
      // TODO: Another is to shrink the feature set size.
      // TODO: Another is to figure out a clever update rule, that only touches the non-zero elements.
      // TODO: The easiest method that will also apply to the neural setting is to do parallel processing. So I should do that first.
      scores.set(i, logScore);
    }
    return Optional.of(scores);
  }

  public BayesianSetsQueryResult query(Collection<String> querySet) {
    ImmutableList.Builder<StringScoreTuple> b = ImmutableList.builder();
    Optional<Vector> scores = getScores(querySet);
    if (!scores.isPresent())
      return null;
    else {
      Vector unpacked = scores.get();
      names.sort((o1, o2) -> {
        Double s2 = unpacked.get(name2Id.get(o2));
        Double s1 = unpacked.get(name2Id.get(o1));
        return (int) signum(s2 - s1);
      });

      for (String name : names)
        b.add(new StringScoreTuple.Builder().setString(name).setScore(unpacked.get(name2Id.get(name))).build());

      return new BayesianSetsQueryResult(b.build(), null);
    }
  }
}
