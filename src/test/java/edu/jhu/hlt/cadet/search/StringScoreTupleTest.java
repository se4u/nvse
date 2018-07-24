package edu.jhu.hlt.cadet.search;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

public class StringScoreTupleTest {

  @Test
  public void testDescendingSort() {
    List<StringScoreTuple> tuples = new ArrayList<>();
    tuples.add(new StringScoreTuple.Builder().setScore(0.0d).setString("zero").build());
    tuples.add(new StringScoreTuple.Builder().setScore(-1.0d).setString("negative").build());
    tuples.add(new StringScoreTuple.Builder().setScore(4.0d).setString("high").build());
    Collections.sort(tuples, StringScoreTuple.descendingScoreComparator());

    assertEquals("high", tuples.get(0).getString());
    assertEquals("zero", tuples.get(1).getString());
    assertEquals("negative", tuples.get(2).getString());
  }
}
