package edu.jhu.hlt.cadet.search;

/**
 * Created by rastogi on 9/9/17.
 */
public enum Algorithm {
  BINARY, GAUSSIAN, NEURAL;

  public static Algorithm fromString(String code) {
    for(Algorithm output : Algorithm.values())
      if(output.toString().equalsIgnoreCase(code))
        return output;
    return null;
  }

}
