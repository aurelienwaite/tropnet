package com.sdl.smert

import breeze.linalg._
import com.sdl.BleuStats
import scala.collection.mutable
import java.io.File

object Sweep {

  def createProjectionMatrix(direction : DenseVector[Float], initial: DenseVector[Float]) = 
    DenseMatrix.vertcat(direction.toDenseMatrix, initial.toDenseMatrix)
  

  def sweepLine(in: DenseMatrix[Float], projection: DenseMatrix[Float]): IndexedSeq[(Float, Int)] = {
    require(projection.rows == 2, "Can only project to lines")
    require(projection.cols == in.rows, "Input matrix and projection matrix must have the same dimension")
    class L(var x: Float, val m: Float, val y: Float, val index: Int) {
      override def toString() = f"x: $x%.4f m: $m%.4f :y $y%.4f i: $index"
    }
    object L {
      def apply(f: DenseVector[Float], i: Int) = new L(Float.NegativeInfinity, f(0), f(1), i)
    }
    val projected = projection * in
    val s = breeze.linalg.argsort(projected(0, ::).t) // s for sorted
    val a: mutable.IndexedSeq[L] = mutable.IndexedSeq.fill(in.cols)(null)
    breeze.linalg.csvwrite(new File("/tmp/projected"), projected.t.map(_.toDouble))
    var j, sIndex = 0
    val end = s.size;
    while (sIndex < end) {
      val l = L(projected(::, s(sIndex)), s(sIndex))
      sIndex += 1
      if (j == 0) {
        a(j) = l
        j += 1
      } else if (!(a(j - 1).m == l.m && l.y <= a(j - 1).y)) {
        if (a(j - 1).m == l.m) 
          j = j - 1
        var exit = false
        while (0 < j && !exit) {
          l.x = (l.y - a(j - 1).y) / (a(j - 1).m - l.m)
          if (a(j - 1).x < l.x)
            exit = true
          else 
            j -= 1
        }
        if (j == 0)
          l.x = Float.NegativeInfinity
        a(j) = l
        j += 1
      }
    }
    for (i <- (0 until j)) yield (a(i).x, a(i).index)
  }

  def sweep(point: DenseVector[Float], directions : DenseMatrix[Float])(in: Tuple2[DenseMatrix[Float], IndexedSeq[BleuStats]]) = {
    val (mat, bs) = in
    for {
      d <- 0 until directions.rows
      projection = createProjectionMatrix(directions(d,::).t, point)
    } yield {
      val intervals = sweepLine(mat, projection)
      val diffs = for (((currInterval, currBS), i) <- intervals.view.zipWithIndex) yield {
        val prevBS = if (i == 0) currBS else intervals(i - 1)._2
        (currInterval, bs(currBS), bs(currBS) - bs(prevBS))
      }
      val res = diffs.toSeq
      res
    }
  }

}
