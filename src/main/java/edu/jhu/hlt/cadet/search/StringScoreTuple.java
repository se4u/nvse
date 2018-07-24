package edu.jhu.hlt.cadet.search;

import java.util.Comparator;

import org.inferred.freebuilder.FreeBuilder;

@FreeBuilder
abstract class StringScoreTuple {
  public abstract String getString();

  public abstract double getScore();

  public static Comparator<StringScoreTuple> descendingScoreComparator() {
    return (sst1, sst2) -> Double.compare(sst2.getScore(), sst1.getScore());
  }

  public static class Builder extends StringScoreTuple_Builder {

  }
}
