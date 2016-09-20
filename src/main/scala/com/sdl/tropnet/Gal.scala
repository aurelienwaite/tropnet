package com.sdl.tropnet

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.PairRDDFunctions
import com.sdl.NBest._
import java.io.File
import com.sdl.smert.SMERT
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.Vectors
import breeze.stats.distributions.RandBasis
import breeze.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.IndexedRow
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import java.io.ObjectOutputStream
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.BufferedWriter
import org.apache.spark.mllib._
import upickle.default._
import java.nio.file.Files
import java.nio.file.Paths
import org.apache.spark.rdd.RDD
import scala.collection.immutable.BitSet

/**
 * Do SVD on an NBest list. Command line arguments
 *
 * 0 - The NBest list directory
 * 1 - A directory on the local system for the V matrix and s vector
 * 2 - The location of the test NBest list directory
 */
object Gal {

  def extractNgrams(in: Array[String], vocab: Map[String, Int]): Seq[Seq[Int]] = {
    val indexed = in.map(vocab.getOrElse(_, 0))
    val ngrams = for (n <- 1 to 5) yield indexed.sliding(n)
    ngrams.flatten.map(_.toVector).toVector
  }

  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("TropNet")//.setMaster("local[2]").set("spark.driver.memory", "10G").set("spark.executor.memory", "10G")
    implicit val sc = new SparkContext(sparkConf)
    sc.setCheckpointDir(".lsa_checkpoint")

    def makeRdd(nbestDir: String) = sc.parallelize(
      (for {
        (nbest, sentIndex) <- loadUCamNBest(new File(nbestDir)).view.zipWithIndex
        (h, hypIndex) <- nbest.zipWithIndex
      } yield (h.hyp.split(" "), (sentIndex, hypIndex))), 500).cache
    val trainRdd = makeRdd(args(0))
    println(trainRdd.count) // This count forces the rdd to be evaluated. Prevents a too many open files issue
    //val testRdd = makeRdd(args(2))
    //println(testRdd.count)
    
    //val numHypsPerSent = (for((nbest, i) <- loadUCamNBest(new File(args(0))).view.zipWithIndex) yield i -> nbest.size).toMap        
    
    type WordsWithSentIndex = (Array[String], (Int, Int))
    
    type Counts = (Int, Map[Int, BitSet])
    def reduceCountsPerSent[T](rdd: PairRDDFunctions[T, Counts]) = rdd.reduceByKey{ case((leftCount, leftMap) , (rightCount, rightMap)) => 
      val unioned = collection.mutable.Map.empty[Int, BitSet]
      for {
        (sentIndex, hypSet) <- (leftMap.toSeq ++ rightMap.toSeq)
      } {
        unioned(sentIndex) = unioned.getOrElse(sentIndex, BitSet.empty) union hypSet
      }
      (leftCount + rightCount, unioned.toMap)
    }
    def countWords(rdd: RDD[WordsWithSentIndex]) = reduceCountsPerSent{
      rdd.flatMap{ case (words, (sentIndex, hypIndex)) => 
        for(word <- words) yield (word, (1, Map(sentIndex -> BitSet(hypIndex))))
      }
    }

    val trainWords = countWords(trainRdd)
    //val testWords = countWords(testRdd)
    //val joinedWords = trainWords join testWords
    
    def collectFeatures[T](rdd: RDD[(T, Counts)]) = rdd.sortBy(_._2._1, ascending=false)
    .filter{ case (_, (_, countsBySent)) => 
      countsBySent.size > 2 && 
        countsBySent.foldLeft(true){ case (atLeast3Hyps, (_, countsByHyp)) => atLeast3Hyps & countsByHyp.size > 2}
    //}.filter{ case (_, (totalCount, _)) =>
    //  totalCount > 1000
    }.collect
    .map{case (feature, (count, _)) => feature -> count}
    val counted = collectFeatures(trainWords)
    println(counted.size)    
    val vocab = Map() ++ (for (((word, _), i) <- counted.view.zipWithIndex) yield word -> (i + 1))
    val pickledVocab = write(vocab)
    Files.write(Paths.get(args(1) + "/vocab.jsn"), pickledVocab.getBytes());
    
    def countNgrams(rdd: RDD[WordsWithSentIndex]) = reduceCountsPerSent{
      rdd.flatMap{ case (words, (sentIndex, hypIndex) ) => 
        for(ngram <- extractNgrams(words, vocab)) yield (ngram, (1, Map(sentIndex -> BitSet(hypIndex))))
      }
    }
    val trainNgrams = countNgrams(trainRdd)
    //val testNgrams = countNgrams(testRdd)
    //val joinedNgrams = trainNgrams join testNgrams
    val ngramsCounted = collectFeatures(trainNgrams)
    val features = Map() ++ (for (((ngram, _), i) <- ngramsCounted.view.zipWithIndex) yield ngram -> (i+5))
    val pickledFeatures = write(features)
    Files.write(Paths.get(args(1) + "/features.jsn"), pickledFeatures.getBytes());
    
    val featuresSize = features.size + 5 
    println(s"feature size: $featuresSize")
    val numHyps = trainRdd.count().toInt
    val vectors = trainRdd.zipWithIndex.map { case ((hyp, _), hypIndex) =>
        val counts = collection.mutable.Map[Int, Double]()
        for (ngram <- extractNgrams(hyp, vocab)) {
          val indexed = features.getOrElse(ngram, ngram.size -1)
          counts(indexed) = 1.0 + counts.getOrElse(indexed, 0.0)
        }
        IndexedRow(hypIndex, Vectors.sparse(featuresSize, counts.toSeq))
    }.persist(StorageLevel.MEMORY_ONLY_2)
    
    trainRdd.unpersist()
    //testRdd.unpersist()
    vectors.checkpoint()

    for (hyp <- vectors.take(10)) println(hyp)
    val rowMat = new IndexedRowMatrix(vectors)

    val svd = rowMat.computeSVD(100, computeU = false)
    println(svd.V.numRows)
    println(svd.V.numCols)
    //println(svd.U.numRows)
    //println(svd.U.numCols)

    def writeObj(filename: String, toWrite: Any) = {
      val out = new BufferedWriter(new FileWriter(filename))
      toWrite match {
        case m: linalg.Matrix => {
          out.write(s"${m.numRows}\t${m.numCols}\n")
          for (i <- 0 until m.numRows)
            out.write((0 until m.numCols).map { j => m(i, j) }.mkString("\t") + "\n")
        }
        case v: linalg.Vector => out.write(v.toArray.mkString("\t") + "\n")
        case _                => sys.error("Unkown type")
      }
      out.close()
    }

    writeObj(args(1) + "/V.obj", svd.V)
    writeObj(args(1) + "/s.obj", svd.s)

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    /*import sqlContext.implicits._
    val df = svd.U.rows.toDF()
    df.show()
    df.write.parquet(args(2))*/

    /*val train = sc.parallelize(input.seq).repartition(100)
    train.persist(StorageLevel.MEMORY_ONLY_2)
    train.checkpoint()

    val trainFeatureVecs = for {
      (ts, _) <- train
      col <- 0 until ts.cols
    } yield {
      val vec = ts(::, col)
      //vec(3) = 0f
      Vectors.dense(vec.toArray.map(_.toDouble))
    }
    val rowMat = new RowMatrix(trainFeatureVecs)
    
    val projection = new DenseMatrix(svd.V.numRows, svd.V.numCols, svd.V.toArray).map(_.toFloat)
    println(projection.cols)
    println(projection.rows)
    for (r <- 0 until projection.rows) {
      for (c <- 0 until projection.cols) print(s"${projection(r, c)} ")
      println()
    }
    val projected = (for((ts, bs) <- train) yield {
      //println(ts.rows + " " + ts.cols)
      //println(projection.rows + " " + projection.cols)
      val projMatrix = projection.t * ts
      //println(ts)
      val wip = ts(3, ::).t.asDenseMatrix
      //println(wip)
      (DenseMatrix.vertcat(projMatrix,wip), bs)
    }).persist(StorageLevel.MEMORY_ONLY_2)
    projected.checkpoint
    
    val generator = SMERT.getGenerator(11)
    val rb = new RandBasis(generator)
    val conf = SMERT.Config(
      initialPoint = DenseVector.fill(NO_OF_UNITS + 1) { 1.0f },
      noOfInitials = 10,
      noOfRandom = 30,
      random = generator,
      activationFactor = None)
    val (point, (newBleu, bp)) = SMERT.doSmert(projected, conf)
  */
  }

}