package edu.jhu.hlt.cadet.search;

import static java.lang.Double.max;
import static java.lang.Double.min;
import static java.lang.System.exit;
import static java.lang.System.out;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.*;

import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.MatrixEntry;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.VectorEntry;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.FlexCompRowMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;

/**
 * Created by rastogi on 6/10/17.
 * Matrix utility functions.
 */
public class MatrixUtils {
  static double sumAll(Matrix mat){
    Iterator<MatrixEntry> iter = mat.iterator();
    double ret = 0;
    while(iter.hasNext()){
      ret += iter.next().get();
    }
    return ret;
  }

  static DenseVector sumRows(Matrix mat) {
    DenseVector ret = (new DenseVector(mat.numColumns())).zero();
    for (MatrixEntry e : mat) {
      int idx = e.column();
      ret.set(idx, ret.get(idx) + e.get());
    }
    return ret;
  }

  @SuppressWarnings("unchecked")
  static <T extends Vector> T sub(T v1, Vector v2) {
    return (T) v1.add(-1, v2);
  }

  static DenseVector zero(int N) {
    Vector v = new DenseVector(N);
    return (DenseVector) v.zero();
  }

  static SparseVector sparseZero(int N) {
    SparseVector v = new SparseVector(N);
    return v.zero();
  }

  static DenseVector constant(int N, Double val) {
    DenseVector v = new DenseVector(N);
    Arrays.fill(v.getData(), val);
    return v;
  }

  static Vector fill(DenseVector v, double val) {
    Arrays.fill(v.getData(), val);
    return v;
  }

  static void sumRows(double[] collector, CompRowMatrix mat, List<Integer> idi){
    Arrays.fill(collector, 0);
    int[] mat_row_ptr = mat.getRowPointers();
    int[] mat_col_ptr = mat.getColumnIndices();
    double[] mat_data = mat.getData();
    for(Integer x: idi)
      for(int j_ = mat_row_ptr[x]; j_ < mat_row_ptr[x+1]; j_++)
        collector[mat_col_ptr[j_]] += mat_data[j_];
  }


  static CompRowMatrix createCRMat(int numRows, int numColumns, int[][] nnz, double[][] data) {
    CompRowMatrix ret = new CompRowMatrix(numRows, numColumns, nnz);
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numColumns; j++) {
        if (data[i][j] != 0)
          ret.set(i, j, data[i][j]);
      }
    }
    return ret;
  }

  static Vector sumRows(CompRowMatrix feat, List<Integer> idi) {
    SparseVector rowIndicator = sparseZero(feat.numRows());
    for (int i : idi) {
      rowIndicator.set(i, 1);
    }
    SparseVector rowSum = new SparseVector(feat.numColumns());
    feat.transMult(rowIndicator, rowSum);
    return rowSum;
  }

  static DenseVector log(DenseVector v) {
    for (VectorEntry e : v)
      e.set(Math.log(e.get()));
    return v;
  }

//  static DenseVector perColumnStd(Matrix mat, Vector columnMeanVector) {
//    DenseVector ret = (new DenseVector(mat.numColumns())).zero();
//    Iterator<MatrixEntry> iter = mat.iterator();
//    while (iter.hasNext()) {
//      MatrixEntry e = iter.next();
//      int idx = e.column();
//      Double value = e.get() - columnMeanVector.get(idx);
//      ret.set(idx, ret.get(idx) + value * value);
//    }
//    Iterator<VectorEntry> vecIter = ret.iterator();
//    Double sqrtNRow = Math.sqrt(mat.numRows());
//    while (vecIter.hasNext()) {
//      VectorEntry e = vecIter.next();
//      e.set(Math.sqrt(e.get()) / sqrtNRow);
//    }
//    return ret;
//  }
//
//  static DenseVector perColumnMean(Matrix mat) {
//    return sumRows(mat).scale(1.0 / mat.numRows());
//  }

  static DenseVector perColumnMeanSansZero(Matrix mat){
    DenseVector ret = zero(mat.numColumns());
    DenseVector nnzPerCol = zero(mat.numColumns());
    for(MatrixEntry e: mat){
      int j = e.column();
      ret.set(j, ret.get(j) + e.get());
      nnzPerCol.set(j, nnzPerCol.get(j) + 1);
    }
    for(int j = 0; j < ret.size(); j++)
      ret.set(j, ret.get(j) / nnzPerCol.get(j));
    return ret;
  }

  static DenseVector perColumnStdSansZero(Matrix mat, Vector columnMeanVector) {
    DenseVector ret = zero(mat.numColumns());
    DenseVector nnzPerCol = zero(mat.numColumns());
    for(MatrixEntry e: mat){
      int j = e.column();
      double tmp = e.get() - columnMeanVector.get(j);
      ret.set(j, ret.get(j) +  (tmp * tmp));
      nnzPerCol.set(j, nnzPerCol.get(j) + 1);
    }
    for(int j = 0; j < ret.size(); j++)
      ret.set(j, Math.sqrt(ret.get(j) / nnzPerCol.get(j)));
    return ret;
  }
  /**
   * Binarize the elements of a column if they are more than factor * STD higher
   * than column's STD
   * 
   * @param feat The feature matrix to binarize
   * @param factor The factor by which we should binarize it.
   * @return Binarized matrix.
   */
  static <T extends Matrix> T binarize(T feat, double factor) {
    @SuppressWarnings("unchecked")
    T output = (T) feat.copy();
//    for(MatrixEntry e: feat){
//      if(e.column() == 1163)
//        out.println(String.valueOf(e.row()) + " " + String.valueOf(e.get()));
//    }
    Vector mean = perColumnMeanSansZero(feat);
    Vector std = perColumnStdSansZero(feat, mean);
    Vector thresh = mean.copy().add(factor, std);
    for (MatrixEntry e : output)
      e.set((e.get() >= thresh.get(e.column())) ? 1 : 0);

//    Set<Integer> zeroColIdi = new HashSet<Integer>(zeroColumns(output));
//    out.println(zeroColIdi.get(0));
//    out.println(zeroColIdi.get(1));
//    Double[] zeroCol0_ = getColumn(feat, zeroColIdi.get(0));
//    Double[] zeroCol1_ = getColumn(feat, zeroColIdi.get(1));
//    Double[] zeroCol0 = getColumn(output, zeroColIdi.get(0));
//    Double[] zeroCol1 = getColumn(output, zeroColIdi.get(1));
    return output;
  }

  static SparseVector pruneZeroCols(SparseVector feat, int newSize, Map<Integer, Integer> old2new){
    int[] index = feat.getIndex();
    double[] data = feat.getData();
    int nnz = 0;
    for(int i = 0; i < index.length; i++)
      if(old2new.containsKey(index[i]))
        nnz++;

    int[] newIndex = new int[nnz];
    double[] newData = new double[nnz];
    int cntr = 0;
    for(int i = 0; i < index.length; i++) {
      if (old2new.containsKey(index[i])) {
        newIndex[cntr] = old2new.get(index[i]);
        newData[cntr] = data[i];
        cntr++;
      }
    }
    return new SparseVector(newSize, newIndex, newData);
  }

  static Map<Integer, Integer> old2New(Matrix feat){
    Set<Integer> zeroColIdi = new HashSet<>(zeroColumns(feat));
    int nCol = feat.numColumns() - zeroColIdi.size();
    int nRow = feat.numRows();
    // HashMap<Integer, Integer> new2old = new HashMap<>();
    Map<Integer, Integer> old2new = new HashMap<>();
    int cntr = 0;
    for(int i = 0; i < feat.numColumns(); i++){
      if(! zeroColIdi.contains(i)) {
        // new2old.put(cntr, i);
        old2new.put(i, cntr);
        cntr++;
      }
    }
    return old2new;
  }

  static CompRowMatrix pruneZeroCols(CompRowMatrix feat, Map<Integer, Integer> old2new){
    int nRow = feat.numRows();
    // Map<Integer, Integer> old2new = old2New(feat);
    int nCol = old2new.size();
    FlexCompRowMatrix mat = new FlexCompRowMatrix(feat);
    FlexCompRowMatrix outMat = new FlexCompRowMatrix(nRow, nCol);
    for(int i = 0; i < nRow; i++)
      outMat.setRow(i, pruneZeroCols(mat.getRow(i), nCol, old2new));
    return new CompRowMatrix(outMat);
  }


  static CompRowMatrix pruneZeroCols(CompRowMatrix feat){
    return pruneZeroCols(feat, old2New(feat));
  }


  static <T extends Matrix> T normalize(T feat) {
    T output = (T) feat.copy();
    Vector mean = perColumnMeanSansZero(feat);
    Vector std = perColumnStdSansZero(feat, mean);
    for (MatrixEntry e : output) {
      double eval = e.get();
      double mu = mean.get(e.column());
      double sigma = std.get(e.column());
      e.set((sigma < 1e-6) ? 0 : ((eval - mu) / sigma));
    }
//    double[] sumOut = MatrixUtils.sumRows(output).getData();
//    OptionalDouble max = Doubles.asList(sumOut).stream().mapToDouble(x->x).max();
//    double sum = MatrixUtils.sumAll(output);
    return output;
  }

  static <T extends Matrix> Double[] getColumn(T mat, int idx){
    int nnz = 0;
    Iterator<MatrixEntry> iter = mat.iterator();
    while(iter.hasNext()){
      MatrixEntry e = iter.next();
      if(idx == e.column()) nnz++;
    }
    Double[] ret = new Double[nnz];
    iter = mat.iterator();
    int cntr = 0;
    while(iter.hasNext()){
      MatrixEntry e = iter.next();
      if(idx == e.column()){
        ret[cntr] = e.get();
        cntr++;
      }
    }
    return ret;
  }

  static <T extends Matrix> T sqrt(T feat) {
    T output = (T) feat.copy();
    Iterator<MatrixEntry> iter = output.iterator();
    while (iter.hasNext()) {
      MatrixEntry e = iter.next();
      e.set(Math.sqrt(e.get()));
    }
    return output;
  }

  static void increment(Matrix mat, int i, int j) {
    mat.set(i, j, mat.get(i, j) + 1);
  }

  static void serializeCSR(ObjectOutputStream oos, CompRowMatrix feat) {
    Integer nRow = feat.numRows();
    Integer nCol = feat.numColumns();
    double[] data = feat.getData();
    Integer nnz = data.length;
    int[] columnIdi = feat.getColumnIndices();
    int[] rowPtr = feat.getRowPointers();
    try {
      oos.writeObject(nRow);
      oos.writeObject(nCol);
      oos.writeObject(nnz);
      oos.writeObject(data);
      oos.writeObject(columnIdi);
      oos.writeObject(rowPtr);
    } catch (Exception e) {
      e.printStackTrace();
      exit(1);
    }

  }

  static <T extends Matrix> List<Integer> zeroColumns(T mat){
    List<Integer> ret = new ArrayList<>();
    boolean[] nzCol = new boolean[mat.numColumns()];
    for(int i = 0; i < nzCol.length; i++) nzCol[i] = false;
    for(MatrixEntry e: mat) if(e.get() != 0) nzCol[e.column()] = true;
    for(int i = 0; i < nzCol.length; i++) if(!nzCol[i]) ret.add(i);
    return ret;
  }

  static CompRowMatrix deserializeCSR(ObjectInputStream is) {
    try {
      Integer nRow = (Integer) is.readObject();
      Integer nCol = (Integer) is.readObject();
      Integer nnz = (Integer) is.readObject();
      double[] data = (double[]) is.readObject();
      int[] columnIdi = (int[]) is.readObject();
      int[] rowPtr = (int[]) is.readObject();
      FlexCompRowMatrix mat = new FlexCompRowMatrix(nRow, nCol);
      for (int i = 0; i < nRow; i++) {
        SparseVector v = new SparseVector(nCol, Arrays.copyOfRange(columnIdi, rowPtr[i], rowPtr[i + 1]),
            Arrays.copyOfRange(data, rowPtr[i], rowPtr[i + 1]));
        mat.setRow(i, v);
      }
      return new CompRowMatrix(mat);
    } catch (Exception e) {
      e.printStackTrace();
      exit(1);
    }
    return null;
  }

  static void printMinMax(String message, Matrix m) {
    double minColSum = 100000;
    double maxColSum = 0;
    for (int j = 0; j < m.numColumns(); j++) {
      double tmpColSum = 0;
      for (int i = 0; i < m.numRows(); i++)
        tmpColSum += m.get(i, j);
      minColSum = min(minColSum, tmpColSum);
      maxColSum = max(maxColSum, tmpColSum);
    }
    out.println(String.format(message + " min=%f, max=%f", minColSum, maxColSum));
  }

  static void sSE(double[] collector, CompRowMatrix mat, List<Integer> idi, double[] mean) {
    Arrays.fill(collector, 0);
    int[] mat_row_ptr = mat.getRowPointers();
    int[] mat_col_ptr = mat.getColumnIndices();
    double[] mat_data = mat.getData();
    int[] nnzPerCol = new int[collector.length];
    for(Integer x: idi){
      for(int j_ = mat_row_ptr[x]; j_ < mat_row_ptr[x+1]; j_++){
        int col = mat_col_ptr[j_];
        nnzPerCol[col]++;
        double delta = (mat_data[j_] - mean[col]);
        collector[col] += (delta * delta);
      }
    }
    int D = idi.size();
    for(int j = 0; j < collector.length; j++){
      if(mean[j] != 0)
        collector[j] += (D - nnzPerCol[j]) * (mean[j] * mean[j]);
    }
  }

//  static List<T> retrieveObjectWithHighestScore(List<T> lobj, List<Double> scores, int k){
//    // TODO: Dont allocate a new array and consume memory un-necessarily.
//    return retrieveObjectWithHighestScore(lobj, (Double[]) scores.toArray(), k);
//  }


  static <T> List<T> retrieveObjectWithHighestScore(List<T> lobj, Double[] scores, int k){
    // 1. Create an indexer array
    Integer[] idi = new Integer[scores.length];
    for(int i = 0; i < idi.length; i++)
      idi[i] = i;
    // 2. Sort the indexer according to scores.
    Arrays.sort(idi, Comparator.comparingDouble(o -> scores[o]));
    // 3. Create a list using the sorted indexer.
    List<T> retList = new ArrayList<>();
    for(int i = idi.length - 1; i >= max(0,idi.length - k); i--)
      retList.add(lobj.get(idi[i]));
    return retList;
  }

}
