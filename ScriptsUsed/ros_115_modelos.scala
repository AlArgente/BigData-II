package main.scala.djgarcia

import java.io.PrintWriter
import java.io.File
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature._

import org.apache.spark.mllib.evaluation.MulticlassMetrics

object runEnsembles {


  def main(arg: Array[String]) {

    //Basic setup
    val jobName = "MLlib ROS 140"

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)
		// val pathHeader = "file:///home/spark/datasets/susy.header"
		val pathHeader = "hdfs://hadoop-master/user/datasets/master/higgs/higgs.header"
    //Load train and test
    // val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
		val pathTrain = "hdfs://hadoop-master/user/datasets/master/higgs/higgsMaster-Train.data"
    val rawDataTrain = sc.textFile(pathTrain)

    // val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
		val pathTest = "hdfs://hadoop-master/user/datasets/master/higgs/higgsMaster-Test.data"
    val rawDataTest = sc.textFile(pathTest) // Se puede añadir ", 10" como parámetro para facilitar a los procesos el uso de ficheros grandes.

    //Load train and test with KeelParser

    val converter = new KeelParser(sc, pathHeader)
    val train = sc.textFile(pathTrain, 10).map(line => converter.parserToLabeledPoint(line)).persist
    val test = sc.textFile(pathTest, 10).map(line => converter.parserToLabeledPoint(line)).persist


		val trainROS = ROS(train, 1.15).persist
		val classInfo = trainROS.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()
    //Class balance
		// val classInfo = train.map(lp => (lp.label, 1L)).reduceByKey(_ + _).collectAsMap()
    // se puede hacer también:
    // val classInfo = train.map(lp => (lp.label, 1L))
    // Aquí SPARK no ha hecho nada aún
    //  .reduceByKey(_ + _) // acción --> Spark trabaja. Se reduce por clave de forma separada, para así tener por un lado los ejemplos de la etiqueta 1, y por otro los de la etiqueta 0, y que sume todos los que hay los de una etiqueta por un lado y los de la otra por otro lado.
    // .collectAsMap() // Se va a todos los reduce y los junta para imprimirlos por pantalla.

		//HME-BD Noise Filter
/*
		val nTrees_hme = 100
		val maxDepthRF_hme = 10
		val partitions_hme = 4

		val hme_bd_model = new HME_BD(train, nTrees_hme, partitions_hme, maxDepthRF_hme, 48151623)

		val hme_bd = hme_bd_model.runFilter()

		hme_bd.persist()
*/
    //Decision tree

    // Train a DecisionTree model.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    var numClasses = converter.getNumClassFromHeader()
    var categoricalFeaturesInfo = Map[Int, Int]()
    var impurity = "gini"
    var maxDepth = 10
    var maxBins = 100

    val modelDT = DecisionTree.trainClassifier(trainROS, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsDT = test.map { point =>
      val prediction = modelDT.predict(point.features) // features s una característica de LabeledPoint
      (point.label, prediction)
    }
    val testAccDT = 1 - labelAndPredsDT.filter(r => r._1 != r._2).count().toDouble / test.count()
    println(s"Test Accuracy DT= $testAccDT")

    //Metrics DT
    val metrics_dt = new MulticlassMetrics(labelAndPredsDT)
    val precision_dt = metrics_dt.precision
    val cm_dt = metrics_dt.confusionMatrix
		val negative_dt = cm_dt.toArray(0) + cm_dt.toArray(2)
		val positive_dt = cm_dt.toArray(1) + cm_dt.toArray(3)
		val TNR_dt = cm_dt.toArray(0) / negative_dt
		val TPR_dt = cm_dt.toArray(3) / positive_dt
		val res_dt = TNR_dt * TPR_dt
		println(s"TNR: $TNR_dt" )
		println(s"TPR: $TPR_dt" )
		println(s"Resultado: $res_dt" )


    //Write Results
    // Para guardar en el cluster
    // val writer_dt = new PrintWriter(new File("/home/spark/results_hmd_filter_dt.txt"))
    val writer_dt = new PrintWriter(new File("/home/x76654048/results_ros_115_dt.txt"))
    writer_dt.write(
      "Precision: " + precision_dt + "\n" +
        "Confusion Matrix " + cm_dt + "\n"
    )
		writer_dt.write("TNR: " + TNR_dt + "\n")
		writer_dt.write("TPR: " + TPR_dt + "\n")
		writer_dt.write("Resultado: " + res_dt + "\n")
    writer_dt.close()




    //Random Forest

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    numClasses =  converter.getNumClassFromHeader()
    categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 100
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    impurity = "gini"
    maxDepth = 10
    maxBins = 100

    val modelRF = RandomForest.trainClassifier(trainROS, numClasses, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPredsRF = test.map { point =>
      val prediction = modelRF.predict(point.features)
      (point.label, prediction)
    }
    val testAccRF = 1 - labelAndPredsRF.filter(r => r._1 != r._2).count.toDouble / test.count()
    println(s"Test Accuracy RF= $testAccRF")

		//Metrics RF
    val metrics_rf = new MulticlassMetrics(labelAndPredsRF)
    val precision = metrics_rf.precision
    val cm = metrics_rf.confusionMatrix
		val negative = cm.toArray(0) + cm.toArray(2)
		val positive = cm.toArray(1) + cm.toArray(3)
		val TNR = cm.toArray(0) / negative
		val TPR = cm.toArray(3) / positive
		val res = TNR * TPR
		println(s"TNR: $TNR" )
		println(s"TPR: $TPR" )
		println(s"Resultado: $res" )


    //Write Results
    // Para guardar en el cluster
    // val writer = new PrintWriter(new File("/home/spark/results_hmd_filter_rf.txt"))
    val writer = new PrintWriter(new File("/home/x76654048/results_ros_115_rf.txt"))
    writer.write(
      "Precision: " + precision + "\n" +
        "Confusion Matrix " + cm + "\n"
    )
		writer.write("TNR: " + TNR + "\n")
		writer.write("TPR: " + TPR + "\n")
		writer.write("Resultado: " + res + "\n")
    writer.close()

    //PCARD

    import org.apache.spark.mllib.tree.PCARD

    val cuts = 10
    val trees = 50

    val pcardTrain = PCARD.train(trainROS, trees, cuts)

    val pcard = pcardTrain.predict(test)

    val labels = test.map(_.label).collect()

    var cont = 0

    for (i <- labels.indices) {
      if (labels(i) == pcard(i)) {
        cont += 1
      }
    }

    val testAcc = cont / labels.length.toFloat

    println(s"Test Accuracy = $testAcc")
    val predsAndLabels = sc.parallelize(pcard).zipWithIndex.map { case (v, k) => (k, v) }.join(test.zipWithIndex.map { case (v, k) => (k, v.label) }).map(_._2)
		/*var predsAndLabels = test.map { point =>
			val prediction = modelRF.predict(point.features)
			(prediction, point.label)
		}*/
		//Metrics PCARD
    val metrics_pc = new MulticlassMetrics(predsAndLabels)
    val precision_pc = metrics_pc.precision
    val cm_pc = metrics_pc.confusionMatrix
		val negative_pc = cm_pc.toArray(0) + cm_pc.toArray(2)
		val positive_pc = cm_pc.toArray(1) + cm_pc.toArray(3)
		val TNR_pc = cm_pc.toArray(0) / negative_pc
		val TPR_pc = cm_pc.toArray(3) / positive_pc
		val res_pc = TNR_pc * TPR_pc
		println(s"TNR: $TNR_pc" )
		println(s"TPR: $TPR_pc" )
		println(s"Resultado: $res_pc" )


    //Write Results
    // Para guardar en el cluster
    // val writer_pc = new PrintWriter(new File("/home/spark/results_hmd_filter_pcard.txt"))
    val writer_pc = new PrintWriter(new File("/home/x76654048/results_ros_115_pcard.txt"))
    writer_pc.write(
      "Precision: " + precision_pc + "\n" +
        "Confusion Matrix " + cm_pc + "\n"
    )
		writer_pc.write("TNR: " + TNR_pc + "\n")
		writer_pc.write("TPR: " + TPR_pc + "\n")
		writer_pc.write("Resultado: " + res_pc + "\n")
    writer_pc.close()

		// Naive Bayes
		val model_nb = NaiveBayes.train(trainROS, lambda = 1.0, modelType = "multinomial")
		val predsAndLabels_nb = test.map { point =>
			val prediction = model_nb.predict(point.features)
			(prediction, point.label)
		}
		//Metrics Bayes
    val metrics_nb = new MulticlassMetrics(predsAndLabels_nb)
    val precision_nb = metrics_nb.precision
    val cm_nb = metrics_nb.confusionMatrix
		val negative_nb = cm_nb.toArray(0) + cm_nb.toArray(2)
		val positive_nb = cm_nb.toArray(1) + cm_nb.toArray(3)
		val TNR_nb = cm_nb.toArray(0) / negative_nb
		val TPR_nb = cm_nb.toArray(3) / positive_nb
		val res_nb = TNR_pc * TPR_pc
		println(s"TNR: $TNR_nb" )
		println(s"TPR: $TPR_nb" )
		println(s"Resultado: $res_nb" )


    //Write Results
    // Para guardar en el cluster
    // val writer_nb = new PrintWriter(new File("/home/spark/results_hmd_filter_bayes.txt"))
    val writer_nb = new PrintWriter(new File("/home/x76654048/results_ros_115_bayes.txt"))
    writer_nb.write(
      "Precision: " + precision_nb + "\n" +
        "Confusion Matrix " + cm_nb + "\n"
    )
		writer_nb.write("TNR: " + TNR_nb + "\n")
		writer_nb.write("TPR: " + TPR_nb + "\n")
		writer_nb.write("Resultado: " + res_nb + "\n")
    writer_nb.close()
  }
	// ROS

	def ROS(train: RDD[LabeledPoint], overRate: Double): RDD[LabeledPoint] = {
		var oversample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

		val train_positive = train.filter(_.label == 1)
		val train_negative = train.filter(_.label == 0)
		val num_neg = train_negative.count().toDouble
		val num_pos = train_positive.count().toDouble

		if (num_pos > num_neg) {
		  val fraction = (num_pos * overRate) / num_neg
		  oversample = train_positive.union(train_negative.sample(withReplacement = true, fraction))
		} else {
		  val fraction = (num_neg * overRate) / num_pos
		  oversample = train_negative.union(train_positive.sample(withReplacement = true, fraction))
		}
		oversample.repartition(train.getNumPartitions)
	}


	// RUS

	def RUS(train: RDD[LabeledPoint]): RDD[LabeledPoint] = {
		var undersample: RDD[LabeledPoint] = train.sparkContext.emptyRDD

		val train_positive = train.filter(_.label == 1)
		val train_negative = train.filter(_.label == 0)
		val num_neg = train_negative.count().toDouble
		val num_pos = train_positive.count().toDouble

		if (num_pos > num_neg) {
		  val fraction = num_neg / num_pos
		  undersample = train_negative.union(train_positive.sample(withReplacement = false, fraction))
		} else {
		  val fraction = num_pos / num_neg
		  undersample = train_positive.union(train_negative.sample(withReplacement = false, fraction))
		}
		undersample
	}
}
