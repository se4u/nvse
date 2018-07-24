package edu.jhu.hlt.cadet.search;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by rastogi on 9/22/17.
 */
public class EntitySentencesAndFeatures {
  public List<String> sentences;
  public List<Integer[]> sentenceFeatures;

  EntitySentencesAndFeatures(List<String> sentences, List<Integer[]> sentenceFeatures){
    this.sentences = sentences;
    this.sentenceFeatures = sentenceFeatures;
  }
}
