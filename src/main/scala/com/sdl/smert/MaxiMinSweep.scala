package com.sdl.smert

import breeze.linalg.DenseMatrix
import com.sdl.BleuStats
import com.sdl.tropnet.Caetano.Neuron
import breeze.linalg.sum
import breeze.linalg.DenseVector
import scala.annotation.tailrec
import scala.collection.mutable.Buffer
import scala.collection.mutable.ArrayBuffer
import java.io.File
import breeze.linalg.max

object MaxiMinSweep {

  def maxiMinSweepLine(toDelete: Range)(activated: DenseMatrix[Float], projection: DenseMatrix[Float], bs: IndexedSeq[BleuStats]) = {

    val deactivated = activated.copy
    deactivated(toDelete, ::) := 0.0f
    //breeze.linalg.csvwrite(new File("/tmp/activated"), activated.map(_.toDouble))
    //breeze.linalg.csvwrite(new File("/tmp/deactivated"), deactivated.map(_.toDouble))
    val projectedActivated = projection * activated
    val projectedDeactivated = projection * deactivated

    val minksum = for (c <- 0 until activated.cols) yield {
      val deactivatedBS = if (sum(projectedDeactivated(::, c)) == 0.0) BleuStats.bad else bs(c).deleteActivations()
      val l = Seq((L(projectedActivated(::, c), c), bs(c)), (L(projectedDeactivated(::, c), c), deactivatedBS)).sortBy(_._1.m)
      val left = l(0)._1
      val right = l(1)._1
      //If both have the same gradient, then pick the one with the lowest y
      if (left.m == right.m) {

        val resorted = l.sortBy(_._1.y)
        //println(resorted)
        //println(resorted.head)
        //println(resorted.head._2.activated)
        //println(resorted.last)
        //println(resorted(0))
        (resorted.head, resorted.head, c)
      } else {
        left.x = (right.y - left.y) / (left.m - right.m)
        /*val update = DenseVector(right.x + 1, 1).t * projection
        val activatedScore = update * activated(::,c)
        val deactivatedScore = update * deactivated(::,c)
        println(left)
        println(right)
        println(right.x)
        println(l(1)._2.activated == 1)
        println(s"activated: $activatedScore, deactivated: $deactivatedScore")*/
        //Swap the lines, because we want the minimum
        (l(1), l(0), c)
      }
    }
    //for (m <- minksum) println(m)
    //println(minksum)

    val sorted = minksum.sortBy(_._2._1.x)
    // A bit hacky, but parallel lines are stored twice, so filter them out
    val filtered = sorted.filter { case (left, right, _) => left != right }
    val initials = for (s <- sorted) yield (s._1, s._3)
    /*println("initials")
    for (i <- initials)
      println(i)
    println("filtered")
    for (f <- filtered)
      println(f)
    sys.exit*/

    //println(filtered.size)

    def update(nbest: DenseMatrix[Float], index: Int, l: L) = {
      nbest(0, index) = l.m
      nbest(1, index) = l.y
      nbest
    }

    /*
     * Converts a sequence of L objects into a Matrix representation of the NBest
     */
    def nbest2Mat(nbest: IndexedSeq[((L, BleuStats), Int)]) = {
      val nbestMat = DenseMatrix.zeros[Float](2, nbest.size)
      val updatedBs = ArrayBuffer.fill(nbest.size)(BleuStats.empty)
      for (((l, bs), i) <- nbest) {
        update(nbestMat, i, l)
        updatedBs(i) = bs
      }
      (nbestMat, updatedBs)
    }

    /*
     * Creates a set of intervals. In each interval a hypothesis is minimal. The output is a set of N-best lists in matrix
     * form with the minimal representation
     */
    val minned = filtered.foldLeft(List((nbest2Mat(initials), Float.NegativeInfinity))) { (prev, s) =>
      val (_, (l, bs), index) = s
      //println(s"$index ${l.x}")
      val (nbest, _) = prev.last
      val updated = nbest._1.copy
      update(updated, index, l)
      //println(updated(::, 67))
      val bsUpdated = nbest._2.updated(index, bs)
      prev :+ ((updated, bsUpdated), l.x)
    }

    /*for (seq <- minned.sliding(2)) seq match {
      case Seq(((m, b), start), ((_, _), end)) => {
        //println(s"${m.cols} ${m.rows} ${b.size} $start $end")
        val checked = if (start.isNegInfinity) end - 2 else start
        val point = DenseVector((checked + end) / 2, 1).t
        val activatedScores = (point * projectedActivated).t
        val deactivatedScores = (point * projectedDeactivated).t
        for ((((a, d), bs), i) <- (activatedScores.toScalaVector() zip deactivatedScores.toScalaVector() zip b).view.zipWithIndex) {
          if ((a > d && bs.activated == 1) || (d > a && bs.activated == 0)) {
            println(s"sviib $i $checked $end $a $d ${bs.activated} ${m(::, i)} ${projectedActivated(::, i)} ${projectedDeactivated(::, i)}")
          }
        }

      }
      case Seq(((m, b), start)) =>
        println(s"${m.rows} ${b.size} $start")
    }*/

    case class I(start: Float, end: Float, index: Int)
    val endI = (Float.PositiveInfinity, Int.MinValue)

    def extractIntervals(swept: Seq[(Float, Int)]) = {
      (for (interval <- (swept :+ endI).sliding(2)) yield interval match {
        //case Seq((Float.NegativeInfinity, _), _) => Seq.empty
        case Seq((start, index), (end, _)) => Seq(I(start, end, index))
        case Seq(_)                        => Seq.empty
      }).flatten
    }

    //for (((_, _), i) <- minned) print(s"$i, ")

    case class IB(start: Float, end: Float, b: BleuStats)
        
    val endMinned = ((DenseMatrix.zeros[Float](0, 0), ArrayBuffer.empty), Float.PositiveInfinity)
    val maximinned = for (seq <- (minned :+ endMinned).sliding(2)) yield seq match {
      case Seq(((mat, bs), start), ((_, _), end)) => {
        val swept = Sweep.sweepLine(mat)
        val intervals = extractIntervals(swept)
        //println(s"$start, $end " + intervals.mkString(", "))
        val filtered = for (interval <- intervals) yield {
          if (interval.start >= start && interval.end < end) Seq(IB(interval.start, interval.end, bs(interval.index))) // interval is contained
          else if (interval.start < start && interval.end > start && interval.end < end) Seq(IB(start, interval.end, bs(interval.index))) //interval starts before
          else if (interval.start >= start && interval.start < end && interval.end >= end) Seq(IB(interval.start, end, bs(interval.index))) // interval ends after
          else if (interval.start < start && interval.end >= end) Seq(IB(start, end, bs(interval.index))) // interval contains 
          else Seq.empty // intervals do not overlap
        }
        val res = filtered.flatten.toSeq
        //for (i <- res) println(i)
        res
      }
      case Seq(((mat, bs), start)) => Seq.empty
    }

    val mmSeq = maximinned.flatten.toSeq

    val mmMerged = mmSeq.tail.foldLeft(ArrayBuffer[IB](mmSeq.head)) { (accum, next) =>
      if (accum.last.b == next.b) {
        val last = accum.remove(accum.length - 1)
        accum += IB(last.start, next.end, next.b)
      } else
        accum += next
    }
    
    /*for (i <- mmMerged)  {
      val mid = if (i.start ==  Float.NegativeInfinity && i.end == Float.PositiveInfinity)
        0
      else if(i.start == Float.NegativeInfinity) 
        i.end - 1
      else if (i.end == Float.PositiveInfinity)
        i.start + 1
      else 
        (i.start + i.end) /2
      val p = DenseVector(mid, 1).t * projection
      println(s"$i $mid ${p.t}")
      val neuron = p.t.copy
      neuron(-1) = 0.0f
      neuron(-2) = 0.0f
      println(neuron)
      val scores = (p*activated).t
      println(scores.toArray.view.zipWithIndex.maxBy(_._1))
    }*/
    
    //println(mmMerged)
    val res = mmMerged.map(i => (i.start, i.b))
    res
  }
  //sys.exit()

  /*@tailrec
    def sweepOverMinned(in: List[((DenseMatrix[Float], IndexedSeq[BleuStats]), Float)], accum: Buffer[(Float, BleuStats)]): Seq[(Float, Int)] =
      in match {
        case Nil => Nil
        case head :: tail => {
          val ((mat, bs), interval) = head
          val swept = Sweep.sweepLine(mat)
          val nextInterval = tail match {
            case Nil         => Float.PositiveInfinity
            case (_, i) :: _ => i
          }
          @tailrec
          def filterSwept(toFilter: List[(Float, Int)], accum: Buffer[(Float, Int)]): Unit = toFilter match {
            case Nil => Unit
            case head :: tail => {
              val (sweepInterval, _) = head
              val sweepNextInterval = tail match {
                case Nil         => Float.PositiveInfinity
                case (e, _) :: _ => e
              }
              if ((sweepInterval >= interval && sweepInterval < nextInterval) ||
                (sweepInterval <= interval && sweepNextInterval > nextInterval)) accum += head
              filterSwept(tail, accum)
            }
          }
          val filtered = Buffer[(Float, Int)]()
          filterSwept(swept.toList, filtered)
          val filteredWithBs = filtered.map {
            case (interval, hypIndex) =>
              (interval, bs(hypIndex))
          }
          //println(s"$interval, $nextInterval")
          //println(swept)
          for (h <- filtered.headOption) {
            val first = (interval, bs(h._2))
            accum append first
            accum ++ filtered.tail
          }
          sweepOverMinned(tail, accum)
        }
      }

    val swept = Buffer[(Float, BleuStats)]()
    sweepOverMinned(minned, swept)
    val merged = swept.tail.foldLeft(Buffer[(Float, BleuStats)](swept.head)) { (accum, next) =>
      if (accum.last._2 != next._2) accum += next
      accum
    }
    val res = merged.toIndexedSeq
    //println(res)
    res
  }*/

}