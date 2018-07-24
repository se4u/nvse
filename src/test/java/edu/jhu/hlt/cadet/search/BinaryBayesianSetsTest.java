package edu.jhu.hlt.cadet.search;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.junit.Before;
import org.junit.Test;

import no.uib.cipr.matrix.sparse.CompRowMatrix;

import static java.lang.System.out;

/**
 * Created by rastogi on 6/10/17.
 */
public class BinaryBayesianSetsTest {
  private List<CompRowMatrix> featList;
  private List<Map<String, Integer>> name2IdList;
  private List<List<Set<String>>> listOfQueries;
  private List<List<List<String>>> expectedResults;

  @Before
  public void setUp() throws Exception {
    CompRowMatrix feat = MatrixUtils.createCRMat(4, 3, new int[][] { { 0 }, { 1 }, { 2 }, { 0, 1 } },
        new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } });
    String[] names = new String[] { "f1", "f2", "f3", "f1f2" };

    Map<String, Integer> name2Id = new HashMap<>();
    for (int i = 0; i < 4; i++) {
      name2Id.put(names[i], i);
    }

    Set<String> query = new HashSet<>();
    query.add("f1");
    query.add("f2");
    assert query.size() == 2;

    List<String> expectedResult = Arrays.asList(new String[] { "f1f2", "f1", "f2", "f3" });

    expectedResults = Arrays.asList(Arrays.asList(expectedResult));
    featList = Arrays.asList(feat);
    name2IdList = Arrays.asList(name2Id);
    listOfQueries = Arrays.asList(Arrays.asList(query));
  }

  @Test
  public void getScores() throws Exception {
  }

  @Test
  public void query() throws Exception {
    for (int i = 0; i < featList.size(); i++) {
      BinaryBayesianSets bs = new BinaryBayesianSets(featList.get(i), name2IdList.get(i));
      for (int j = 0; j < listOfQueries.get(i).size(); j++) {
        Set<String> query = listOfQueries.get(i).get(j);
        List<String> expectedResult = expectedResults.get(i).get(j);
        BayesianSetsQueryResult qr = bs.query(query);
        Map<Integer, Double> lm = qr.lm;
        List<StringScoreTuple> producedResult = qr.eScores;
        out.println(producedResult.toString());
        out.println(lm);
        // assertEquals(expectedResult, producedResult);
      }
    }
  }
}