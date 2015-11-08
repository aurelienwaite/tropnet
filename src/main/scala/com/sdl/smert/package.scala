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
  
  type F = Float
  
  val F = Float
  
  type Vector = DenseVector[F]
  
  type Matrix = DenseMatrix[F]
  
  sealed trait StringToF[T]{def toF(in : String) : T}
  implicit object StringToDouble extends StringToF[Double] {def toF(in : String) = in.toDouble}
  implicit object StringToFloat extends StringToF[Float] {def toF(in : String) = in.toFloat}
    
  def stringToVec(in : String)(implicit parser : StringToF[F]) =  DenseVector(in.split(",").map(parser.toF(_))) 
   
  def sweepLine(in: Matrix, projection: Matrix): IndexedSeq[(F, Int)] = {
    require(projection.rows == 2, "Can only project to lines")
    require(projection.cols == in.rows, "Input matrix and projection matrix must have the same dimension")
    class L(var x: F, val m: F, val y: F, val index: Int)

    object L {
      def apply(f: DenseVector[F], i: Int) = new L(F.NegativeInfinity, f(0), f(1), i)
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
          if (j == 0) l.x = F.NegativeInfinity
          a(j) = l
          j += 1
        }
      }
    }
    for (i <- (0 until j)) yield (a(i).x, a(i).index)
  }
  
  implicit def nbestToMatrix(in : NBest) : Matrix = {
    require(in.size > 0, "NBest needs to have at least one element")
    val fVecDim = in(0).fVec.size
    val buf = new ArrayBuffer[F]
    for(hyp <- in) {
      require(hyp.fVec.size == fVecDim, "Feature vecs must be of the same dimension")
      buf ++= hyp.fVec.map { _.asInstanceOf[F] * -1.0.asInstanceOf[F]}
    }
    new DenseMatrix(fVecDim, in.size, buf.toArray)
  }

}