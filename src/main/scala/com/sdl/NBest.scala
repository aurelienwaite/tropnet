package com.sdl

import java.io.File
import scala.io.Source
import java.util.zip.GZIPInputStream
import java.io.FileInputStream

import resource._

/**
 * @author aaw35
 */
object NBest {

  type NBest = IndexedSeq[Hypothesis]

  type NBestSet = Seq[NBest]

  /**
   * Filename to Index
   */
  def f2i(f: File) = f.getName.split("\\.")(0).toInt

  def loadUCamNBest(dir: File): NBestSet = {
    val in = for {
      nbestFile <- dir.listFiles.toSeq.sortWith(f2i(_) < f2i(_))
    } yield {
      if (nbestFile.getName.endsWith(".gz"))
        Source.fromInputStream(new GZIPInputStream(new FileInputStream(nbestFile)))
      else Source.fromFile(nbestFile)
    }
    loadUCamNBest(in.toList)
  }

  def loadUCamNBest(in: List[Source]): Stream[NBest] = {
    in match {
      case Nil => Stream.empty
      case head :: tail => {
        val nBest = loadUCamNBest(head)
        nBest #:: loadUCamNBest(tail)
      }
    }
  }

  def loadUCamNBest(in: Source): NBest = managed(in) acquireAndGet { in =>
    for (line <- in.getLines.toIndexedSeq) yield {
      val fields = line.split("\t")
      val fVec = fields(1).split(",").map(_.toDouble).toIndexedSeq
      val sbleu = fields(3).toDouble
      Hypothesis(fields(0), fVec, BleuStats(fields(2)), sbleu)
    }
  }

}