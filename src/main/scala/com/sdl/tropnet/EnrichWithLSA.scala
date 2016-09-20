package com.sdl.tropnet

import java.io.File
import com.sdl.NBest._
import java.io.BufferedReader
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import upickle.default._
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg
import breeze.linalg.SparseVector
import breeze.linalg.VectorBuilder

object EnrichWithLSA extends App{
  
  val matrixSource = Source.fromFile(args(1)).getLines
  val (rows, cols) = matrixSource.next().split("\\s").map(_.toInt) match {
    case Array(rows: Int, cols: Int) => (rows, cols)
    case _ => sys.error("Unkown dim format")
  }
  println(s"$rows $cols")
  val data = (for(line <- matrixSource) yield line.split("\\s").toIndexedSeq.map(_.toDouble)).toIndexedSeq
  val wordEmbeddings = DenseMatrix.tabulate(rows, cols)((i,j) => data(i)(j))
  val singularValues = DenseVector(Source.fromFile(args(2)).getLines.next.split("\\s").map(_.toDouble))
  val inverted = linalg.diag(singularValues.map(1/_))
  println(wordEmbeddings)
  println(singularValues)
  val mapping = inverted * wordEmbeddings.t
  println(mapping)
  
  val vocab = read[Map[String, Int]](Source.fromFile(args(3)).mkString)
  val features = read[Map[Seq[Int], Int]](Source.fromFile(args(4)).mkString)
  val nbests = loadUCamNBest(new File(args(0))).iterator
  val transformed = for (nbest <- nbests) yield {
    for (hyp <- nbest) yield {
      val counts = collection.mutable.Map[Int, Double]()
      for {
         ngram <- Gal.extractNgrams(hyp.hyp.split("\\s"), vocab)
         indexed = features.getOrElse(ngram, ngram.size - 1)
       } counts(indexed) = counts.getOrElse(indexed, 0.0) + 1.0
       val vecBuilder = new VectorBuilder[Double](features.size + 5)
       for((indexed, count) <- counts) vecBuilder.add(indexed, count)
       val vec = vecBuilder.toSparseVector(alreadySorted = false, keysAlreadyUnique = true)
       val projected = (mapping * vec)
       hyp.copy(fVec = hyp.fVec ++ (projected :* -1.0).toArray)
    }
  }
  saveUCamNBest(transformed, new File(args(0) + "_LSA_NGRAMS_REPRO"))  
}