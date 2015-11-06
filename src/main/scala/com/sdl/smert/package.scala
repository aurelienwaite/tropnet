package com.sdl

import com.sdl.NBest._
import breeze.linalg._
import scala.collection._
import scala.collection.mutable.ArrayBuffer
import scala.language.implicitConversions

/**
 * @author aaw35
 */
package object smert {

  def sweepLine(in: DenseMatrix[Float], projection: DenseMatrix[Float]): IndexedSeq[(Float, Int)] = {
    require(projection.rows == 2, "Can only project to lines")
    require(projection.cols == in.rows, "Input matrix and projection matrix must have the same dimension")
    class L(var x: Float, val m: Float, val y: Float, val index: Int)

    object L {
      def apply(f: DenseVector[Float], i: Int) = new L(Float.NegativeInfinity, f(0), f(1), i)
    }

    val projected = projection * in
    val s = argsort(projected(0, ::).t) // s for sorted
    val a: mutable.IndexedSeq[L] = mutable.IndexedSeq.fill(in.cols)(null)
    var j = 0
    for (sIndex <- s) {
      val l = L(projected(::, sIndex), sIndex)
      if (j == 0) {
        a(j) = l
        j += 1
      } else {
        if (a(j - 1).m == l.m && l.y > a(j - 1).y) j -= 1
        else {
          var exit = false
          while (0 < j && !exit) {
            l.x = (l.y - a(j - 1).y) / (a(j - 1).m - l.m)
            exit = a(j - 1).x < l.x
            if (!exit) j -= 1
          }
          if (j == 0) l.x = Float.NegativeInfinity
          a(j) = l
          j += 1
        }
      }
    }
    for (i <- (0 until j)) yield (a(i).x, a(i).index)
  }
  
  implicit def nbestToMatrix(in : NBest) : DenseMatrix[Float] = {
    require(in.size > 0, "NBest needs to have at least one element")
    val fVecDim = in(0).fVec.size
    val buf = new ArrayBuffer[Float]
    for(hyp <- in) {
      require(hyp.fVec.size == fVecDim, "Feature vecs must be of the same dimension")
      buf ++= hyp.fVec.map { _.toFloat * -1.0f}
    }
    new DenseMatrix(fVecDim, in.size, buf.toArray)
  }

}