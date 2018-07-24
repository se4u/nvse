package edu.jhu.hlt.cadet.search;

import static java.lang.System.out;
import static org.junit.Assert.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.CompRowMatrix;

/**
 * Created by rastogi on 6/24/17.
 */
public class MatrixUtilsTest {
  private Matrix m, m1, m2;
  private Vector v1, v2;
  private CompRowMatrix cr1, cr2, cr3;
  private double[][] cr1Data, cr2Data, cr3Data;

  @Before
  public void setUp() throws Exception {
    m = new DenseMatrix(new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } });
    m1 = new DenseMatrix(new double[][] { { 1, 0, 1 }, { 0, 0, 2 } });
    m2 = new DenseMatrix(new double[][] { { 1, 0, 0 }, { 0, 2, 0 } });
    v1 = new DenseVector(new double[] { 1, 0, 1 });
    v2 = new DenseVector(new double[] { 0, 0, 2 });
    cr1Data = new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } };
    cr1 = MatrixUtils.createCRMat(4, 3, new int[][] { { 0 }, { 1 }, { 2 }, { 0, 1 } }, cr1Data);

    cr2Data = new double[][] { { 4, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } };
    cr2 = MatrixUtils.createCRMat(4, 3, new int[][] { { 0 }, { 1 }, { 2 }, { 0, 1 } }, cr2Data);

    cr3Data = new double[][] { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 }, { 0, 1, 0 } };
    cr3 = MatrixUtils.createCRMat(4, 3, new int[][] { { 0 }, { 1 }, { 2 }, { 0, 1 } }, cr3Data);

  }

  public void eq(Vector v, double[] gold) {
    for (int i = 0; i < v.size(); i++)
      assertEquals(gold[i], v.get(i), 1e-10);
  }

  public void eq(Matrix m, double[][] g) {
    eq(m, g, 1e-10);
  }

  public <T extends Object> void eq(List<T> l1, List<T> l2){
    assertEquals(l1.size(), l2.size());
    for(int i =0; i < l1.size(); i++)
      assertEquals(l1.get(i), l2.get(i));
  }

  public void eq(Matrix m, double[][] gold, double delta) {
    for (int i = 0; i < m.numRows(); i++)
      for (int j = 0; j < m.numColumns(); j++)
        assertEquals(m.get(i, j), gold[i][j], delta);
  }

  @Test
  public void sumRows() throws Exception {
    eq(MatrixUtils.sumRows(m), new double[] { 2, 2, 1 });
  }

  @Test
  public void sub() throws Exception {
    eq(MatrixUtils.sub(v1, v2), new double[] { 1, 0, -1 });
  }

  @Test
  public void zero() throws Exception {
    eq(MatrixUtils.zero(3), new double[] { 0, 0, 0 });
  }

  @Test
  public void sparseZero() throws Exception {
    eq(MatrixUtils.sparseZero(3), new double[] { 0, 0, 0 });
  }

  @Test
  public void constant() throws Exception {
    eq(MatrixUtils.constant(3, 1.), new double[] { 1, 1, 1 });
  }

  @Test
  public void fill() throws Exception {
    DenseVector v = MatrixUtils.zero(3);
    MatrixUtils.fill(v, 1);
    eq(MatrixUtils.constant(3, 1.), v.getData());
  }

  @Test
  public void sumRowsCRByIndex() throws Exception {
    eq(MatrixUtils.sumRows(cr1, Arrays.asList(new Integer[] { 0, 1 })), new double[] { 1, 1, 0 });
  }

  @Test
  public void log() throws Exception {
    eq(MatrixUtils.log(MatrixUtils.sumRows(m)), new double[] { Math.log(2), Math.log(2), 0 });
  }

  @Test
  public void perColumnStd() throws Exception {
    DenseVector v = MatrixUtils.perColumnMeanSansZero(cr2);
    eq(v, (new double[] { 2.5, 1.0, 1.0 }));
    DenseVector s = MatrixUtils.perColumnStdSansZero(cr2, v);
    eq(s, new double[] { 1.5, 0, 0 });
  }

  @Test
  public void normalize() throws Exception {
    assertEquals(0, MatrixUtils.sumAll(MatrixUtils.normalize(cr2)), 1e-8);
  }

  @Test
  public void pruneZeroCols() throws Exception {
    DenseMatrix ret = new DenseMatrix(MatrixUtils.pruneZeroCols(cr3));
    eq(ret, new double[][]{{ 0 }, {  1}, {  0 }, {  1}});
  }
  @Test
  public void binarize() throws Exception {
    Matrix actual = MatrixUtils.binarize(m, 1);
    eq(actual, new double[][] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 0 } });
  }

  @Test
  public void increment() throws Exception {
    Matrix m = this.m.copy();
    MatrixUtils.increment(m, 0, 0);
    assertEquals(m.get(0, 0), 2, 1e-10);
  }

  @Test
  public void deserializeCSR() throws Exception {
    try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
      try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
        MatrixUtils.serializeCSR(oos, cr1);
        byte[] barr = baos.toByteArray();
        eq(MatrixUtils.deserializeCSR(new ObjectInputStream(new ByteArrayInputStream(barr))), cr1Data);
      }
    }
  }


  @Test
  public void retrieveObjectWithHighestScore() throws Exception {
    List<String> ret = MatrixUtils.retrieveObjectWithHighestScore(
      Arrays.asList(new String[]{"0.3", "0.2", "0.7", "0.1"}),
      new Double[]{0.3, 0.2, 0.7, 0.1},
      2);
    out.println(ret);
    eq(ret, Arrays.asList(new String[]{"0.7", "0.3"}));
  }
}