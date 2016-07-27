package com.sdl

case class BleuStats(hits: Seq[(Int, Int)], refLength: Int, activated : Int, numHyps : Int = 1) {

  def +(implicit o: BleuStats) = new BleuStats(hits.zip(o.hits).map {
    t => (t._1._1 + t._2._1, t._1._2 + t._2._2)
  },
    refLength + o.refLength,
    activated + o.activated,
    numHyps + o.numHyps)

  def -(implicit o: BleuStats) = new BleuStats(hits.zip(o.hits).map {
    t => (t._1._1 - t._2._1, t._1._2 - t._2._2)
  },
    refLength - o.refLength,
    activated - o.activated,
    numHyps - o.numHyps)

  def computeBleu(activationScalaingFactor: Option[Double] = None) = {
    val logPrec = hits.map(h => math.log(h._1.toDouble / h._2.toDouble)).reduce(_ + _)
    val logBrev = math.min(0.0, 1 - refLength.toDouble / hits(0)._2.toDouble);
    val scaling = 1.0 / BleuStats.MAX_ORDER.toDouble;
    val activationTerm = activationScalaingFactor.map{s =>
      Math.log(activated.toDouble / numHyps.toDouble) * s
    } getOrElse(0.0)
    //println(math.exp(logPrec * scaling + logBrev) + " " + activated.toDouble / numHyps.toDouble + " " + activationTerm)
    val bleu = (math.exp(logPrec * scaling + logBrev) + activationTerm , math.exp(logBrev));
    bleu
  }
  
  override def toString() = {
    val hitStrings = for (hit <- hits) yield s"${hit._1}/${hit._2}"
    (hitStrings :+ refLength.toString).mkString("; ")
  }
  
  def deleteActivations() = 
    BleuStats(hits, refLength, 0, numHyps)
  
}

object BleuStats {
  val MAX_ORDER = 4

  val empty = new BleuStats(for (i <- 0 until MAX_ORDER) yield (0, 0), 0, 0)

  val bad = new BleuStats(for (i <- 0 until MAX_ORDER) yield (0, 999), 999, 0)
  
  def apply(bsString: String) = {
    val fields = bsString.split("; ").map(_.split("/").map(_.toInt))
    val bleuHits = for (i <- 0 until BleuStats.MAX_ORDER) yield Tuple2(fields(i)(0), fields(i)(1))
    new BleuStats(bleuHits, fields(BleuStats.MAX_ORDER)(0), 1)
  }

}
