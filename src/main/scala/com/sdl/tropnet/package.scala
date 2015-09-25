package com.sdl

import java.io.File
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.language.implicitConversions

import resource._


package object tropnet{

  import java.io.FileInputStream

  type NBest = IndexedSeq[Hypothesis]

  type NBestSet = Seq[NBest]

  val formatter = new java.text.DecimalFormat("0.######")

  class Dot[T](v1: Seq[T])(implicit n: Numeric[T]) {
    import n._ // import * operator
    def dot(v2: Seq[T]) = {
      require(v1.size == v2.size)
        (v1 zip v2).map{ Function.tupled(_ * _)}.sum
    }
  }

  implicit def toDot[T: Numeric](v1: Seq[T]) = new Dot(v1)


  /**
    * Filename to Index 
    */
  def f2i(f : File) =  f.getName.split("\\.")(0).toInt


  def loadUCamNBest(dir : File) : NBestSet = {
    val in = for {nbestFile <- dir.listFiles.toSeq.sortWith(f2i(_) < f2i(_))
    } yield {
      if (nbestFile.getName.endsWith(".gz"))
        Source.fromInputStream(new GZIPInputStream(new FileInputStream(nbestFile)))
      else Source.fromFile(nbestFile)
    }
    loadUCamNBest(in.toList)
  }

  def loadUCamNBest(in : List[Source]) : Stream[NBest]= {
    in match {
      case Nil => Stream.empty
      case head :: tail => {
        val nBest  =    managed(head) acquireAndGet{in =>
          for(line <- in.getLines.toIndexedSeq) yield {
            val fields = line.split("\t")
            val fVec = fields(1).split(",").map(_.toDouble).toIndexedSeq
            val sbleu = fields(3).toDouble
            Hypothesis(fields(0), fVec, BleuStats(fields(2)), sbleu)
          }
        }
        nBest #:: loadUCamNBest(tail)
      }
    }
  }



}

