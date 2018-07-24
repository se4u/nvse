package edu.jhu.hlt.cadet.search;

import java.util.Collection;
import java.util.List;

/**
 * Created by rastogi on 9/9/17.
 */
public interface BayesianSets {
  BayesianSetsQueryResult query(Collection<String> querySet);
}
