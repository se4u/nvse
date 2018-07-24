package edu.jhu.hlt.cadet.search;

import java.util.List;
import java.util.Map;

/**
 * Created by rastogi on 9/22/17.
 */
public class BayesianSetsQueryResult {
  public List<StringScoreTuple> eScores;
  public Map<Integer, Double> lm;

  BayesianSetsQueryResult(List<StringScoreTuple> eScores, Map<Integer, Double> lm){
    this.eScores = eScores;
    this.lm = lm;
  }
}
