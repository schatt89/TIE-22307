package assignment19

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions.{window, column, desc, col}


import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.Column
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.{ArrayType, StringType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.functions.{count, sum, min, max, asc, desc, udf, to_date, avg, mean, stddev}

import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.functions.array
import org.apache.spark.sql.SparkSession

import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}




import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{KMeans, KMeansSummary, KMeansModel}
//Origin 
//import org.apache.spark.mllib.linalg.Vector

//MK 20/26 
import org.apache.spark.ml.linalg.Vector


import java.io.{PrintWriter, File}


//import java.lang.Thread
import sys.process._


import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection.immutable.Range

object assignment {
  // Suppress the log messages:
  Logger.getLogger("org").setLevel(Level.OFF)
                       
  val spark = SparkSession.builder()
                          .appName("assignment19")
                          .config("spark.driver.host", "localhost")
                          .master("local")
                          .getOrCreate()
 
                      
  val dataK5D2: DataFrame = spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D2.csv")
                       

  
  val dataK5D3: DataFrame =  spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D3.csv")
                       

  
  val dataK5D3WithLabels: DataFrame =  spark.read
                       .format("csv")
                       .option("header", "true")
                       .option("inferSchema", "true")
                       .csv("data/dataK5D3.csv")
  
  val toDouble = udf[Double, String]( _.toDouble)
  
  // TASK 1
                       
                      
  
  def task1(df: DataFrame, k: Int) : Array[(Double, Double)]  = {
    //Load data to dataFrame, remove header line and change the a/b column variable type to double
    val data = df.drop("LABEL")
    .withColumn("a", toDouble(df("a")))
    .withColumn("b", toDouble(df("b")))
    .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
    //load data to the memory 
    data.cache()
    data.show(10)
  
  
    // Statistical analysis: Find minimum and maximum value in column a from the whole DataFrame
    //  val statArrayA = data.select(count("a"), min("a"), max("a"), mean("a"), stddev("a")).collect()
    //  println(s"Count, Min, Max and Std of column 'a'")
    //  statArrayA.foreach(println)
  
    //Another way: statistical analysis of column 'a'  
    println(s"\n 	Statistical analysis of column 'a'")
    data.select(count("a"), min("a"), max("a"), mean("a"), stddev("a")).show()
    
   
    //Statistical analysis of column 'b'  
    println(s"\n 	Statistical analysis of column 'b'")
    data.select(count("b"), min("b"), max("b"), mean("b"), stddev("b")).show()
  
                 
    //Create a VectorAssembler for mapping input column "a" and "b" to "features" 
    //This step needed because all machine learning algorithms in Spark take as input a Vector type  
    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("a", "b"))
      .setOutputCol("features")
    
    
    //Another solution: Performs transform and drops if null value in a and b columns
    //val transformedData = vectorAssembler.transform(data.na.drop(Array("a", "b")))

      
    //Perform pipeline with sequence of stages to process and learn from data 
    val transformationPipeline = new Pipeline()
        .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data)
    val transformedTraining = pipeLine.transform(data)

    
    
    // Create a k-means object and fit the transformedTraining to get a k-means object
    //     set parameters: k = 5, number of cluster
    //     k is randomly assigned within the data                : 
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L)

      
      
    // train the model
    val kmModel: KMeansModel = kmeans.fit(transformedTraining)

    
    
   //5 k-means cluster centroids of vector data type converted to array as return values 
    val centers = kmModel.clusterCenters
      .map(x => x.toArray)
      .map{case Array(f1,f2) => (f1,f2)}
    
    println(s"\n Number of centroids = ${centers.length} \n ") 
    return centers

    
  }
  
  println("\n Task 1: k-means clustering for two dimentional data: dataK5D2.csv")
  val answer1 = task1(dataK5D2, 5)
  answer1.foreach(println)
  

  // TASK 2
  
  
  
  def task2(df: DataFrame, k: Int) : Array[(Double, Double, Double)] = {
    
    //Load data to dataFrame, remove header line  
    val data = df.drop("LABEL")
    .withColumn("a", toDouble(df("a")))
    .withColumn("b", toDouble(df("b")))
    .withColumn("c", toDouble(df("c")))
    .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
    data.printSchema()

    //load data to the memory 
    data.cache()
    data.show(10)
  
//    //Statistical analysis of column 'a'  
//    println(s"\n 	Statistical analysis of column 'a'")
//    data.select(count("a"), min("a"), max("a"), mean("a"), stddev("a")).show()
//    
//   
//    //Statistical analysis of column 'b'  
//    println(s"\n 	Statistical analysis of column 'b'")
//    data.select(count("b"), min("b"), max("b"), mean("b"), stddev("b")).show()
//    
//    //Statistical analysis of column 'c'  
//    println(s"\n 	Statistical analysis of column 'c'")
//    data.select(count("c"), min("c"), max("c"), mean("c"), stddev("c")).show()
//    
   
//    //Another way of statistical analysis 
//    val statC: DataFrame = data
//        .select(
//          count("c").alias("count_C"),
//          min("c").alias("minC"),
//          max("c").alias("maxC"),
//          mean("c").alias("meanC"),
//          stddev("c").alias("stddev"))
//        .selectExpr(
//          "count_C",
//          "minC",
//          "maxC",
//          "meanC",
//          "stddev")
//       // "(maxC - minC) /(maxC -minC)")
//        
//   statC.show()

   // Another way to show the list of descriptive statistics 
    println("\n Summary of descriptive statistics ---")
    data.describe().show()

   //The statistical analysis shows that column 'c' standard deviation too high as 273.63044184094457
   // it need to be scaled down, otherwise variance sensitive k-means can compute entirely on the basisi of the 'c' 
 
   //import org.apache.spark.mllib.linalg.Vectors
   import org.apache.spark.ml.linalg.Vectors
   import org.apache.spark.ml.feature.MinMaxScaler
   import org.apache.spark.sql.functions.udf
 
   val vectorizeCol = udf( (v:Double) => Vectors.dense(Array(v)) )        
   
   //MinMaxScaler can be used to transform the 'c' column to be scale down  
   val df1 = data.withColumn("cVec", vectorizeCol(data("c")))    
   val scaler = new MinMaxScaler()
       .setInputCol("cVec")
       .setOutputCol("cScaled")
       .setMax(1)
       .setMin(-1)
 

    //scaler.fit(df1).transform(df1).show
    
    val data2 = scaler
       .fit(df1)
       .transform(df1)
    
    //data2.describe().show()
    //data2.show()       
      
 
    
    //Create a VectorAssembler for mapping input column "a", "b" and "c" to "features"
    //This step needed because all machine learning algorithms in Spark take as input a Vector type  
    val vectorAssembler = new VectorAssembler()
        .setInputCols(Array("a", "b", "cScaled"))
        .setOutputCol("features")
    
    
    //Perform pipeline with sequence of stages to process and learn from data 
    val transformationPipeline = new Pipeline()
        .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data2)
    val transformedTraining = pipeLine.transform(data2)
    
    println("\n Features after scaling down ---")
    transformedTraining.show
    //transformedTraining.describe().show()
    
      
    // Create a k-means object and fit the transformedTraining to get a k-means object
    //     set parameters: k = 5, number of cluster
    //     k is randomly assigned within the data  
    val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
    
    // train the model
    val kmModel: KMeansModel = kmeans.fit(transformedTraining)
    
    println("\n Summary of k-mean clustinering with prediction ---")
    kmModel.summary.predictions.show
    
 
     
    //5 k-means cluster centroids of vector data type converted to array as return values
    val centers = kmModel.clusterCenters.map(x => x.toArray)
        .map{case Array(f1,f2,f3) => (f1,f2,f3)}
    
    println(s"\n Number of centroids = ${centers.length} \n ") 
    

    // Following codes are to make f3 value in the centroids to be scaled up 
    val spark = SparkSession.builder.getOrCreate()
    import spark.implicits._

    val df5 = spark.createDataFrame(centers)
        .toDF("f1", "f2", "f3")
  
    //df5.show()

     // MinMaxScaler takes only vector type, thus f3 transforms to vector from array
     val vectorizeCol2 = udf( (v:Double) => Vectors.dense(Array(v)) )        
   
     //MinMaxScaler can be used to transform the 'f3' column to be scale up  
     val df7 = df5.withColumn("f3Vec", vectorizeCol2(df5("f3")))    
     val scaler2 = new MinMaxScaler()
         .setInputCol("f3Vec")
         .setOutputCol("f3ScaledBack")
         .setMax(991.9577)
         .setMin(9.5387)
 
     println("\n Summary of k-mean clustinering predictions ---")
     //scaler2.fit(df7).transform(df7).show
    
   
      val center2 = scaler2
         .fit(df7)
         .transform(df7)
    
      //center2.describe().show()
      //center2.show()    

      val centersDF = center2.select(col("f1"), col("f2"), col("f3ScaledBack"))
      
      println("\n Centroids after scaling back ---")
      centersDF.show()
      
      //Convert DataFrame to Array  
      val centersArray = centersDF.select("f1", "f2", "f3ScaledBack").collect()
          .map(each => (each.getAs[Double]("f1"), each.getAs[Double]("f2"), each.getAs[Double]("f3ScaledBack")))
          .toArray

    return centersArray
 
  }
  
  println("\n Task 2: k-means clustering for three dimentional data: dataK5D3.csv")
  val answer2 = task2(dataK5D3, 5)
  answer2.foreach(println)


  // TASK 3
  
  def task3(df: DataFrame, k: Int) : Array[(Double, Double)] = {
    // Creating label in binary integer (0 and 1) for future transform of the LABEL to use with ML
    val df_convert = spark.createDataFrame(Seq((0, "Ok"), (1, "Fatal"))).toDF("id", "LABEL")  
     
    df.show()
    df.printSchema()
    
    // Maps a string column of labels to an ML column of label indices
    val indexer = new StringIndexer()
    .setInputCol("LABEL")
    .setOutputCol("lid")
      
    val df1 = indexer
    .fit(df)
    .transform(df)
   
 
    df1.show()
    df1.printSchema()
    
    // Load data to dataFrame, remove header line and cast the a/b column variable type to double
    // Drop LABEL column, but cast label ids (lid) to Double
    val data = df1.drop("LABEL")
                  .selectExpr("cast(a as Double) a", "cast(b as Double) b", "cast(lid as Double) label")
                  .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
   
    data.printSchema()
    
    //load data to the memory 
    data.cache()
    data.show(10)
    
    
    // Create VectorAssembler for mapping input column "a", "b" and "label" to "features"
    // This step needed because ML algorithms in Spark take as input a Vector type
    val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("a", "b", "label"))
    .setOutputCol("features")
    
    // Perform pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline()
    .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data)
    val transformedTraining = pipeLine.transform(data)
    
    // Create a k-means object and fit the transformedTraining to get a k-means object
    val kmeans = new KMeans()
    .setK(k)
    .setSeed(1L)
    val kmModel = kmeans.fit(transformedTraining)
    val centers = kmModel.clusterCenters
    .map(x => x.toArray)
    .map{case Array(f1,f2,f3) => (f1,f2,f3)}
    // changed from 0.5 to 0.43 in order to get two centers  
    .filter(x => (x._3 > 0.43))
    .map{case (f1,f2,f3) => (f1,f2)}
    
    return centers
  }

  println("Task 3:")
  val answer3 = task3(dataK5D3WithLabels, 5)
  answer3.foreach(println)
  

  // Parameter low is the lowest k and high is the highest one.
  def task4(df: DataFrame, low: Int, high: Int): Array[(Int, Double)]  = {
    // Creating label in binary integer (0 and 1) for future transform of the LABEL to use with ML
    val df_convert = spark.createDataFrame(Seq((0, "Ok"), (1, "Fatal"))).toDF("id", "LABEL")

    // Maps a string column of labels to an ML column of label indices
    val indexer = new StringIndexer()
    .setInputCol("LABEL")
    .setOutputCol("lid")
      
    val df1 = indexer
    .fit(df)
    .transform(df)
    
    // Load data to dataFrame, remove header line and cast the a/b column variable type to double
    // Drop LABEL column, but cast label ids (lid) to Double
    val data = df1.drop("LABEL").selectExpr("cast(a as Double) a", "cast(b as Double) b", "cast(lid as Double) label")
                  .na.drop("any")  // Drop any rows in which any value is null in a/b column 
  
   
    data.printSchema()
    
    //load data to the memory 
    data.cache()
    data.show(10)
    
    // Create a VectorAssembler for mapping input column "a", "b" and "c" to "features"
    // This step needed because all machine learning algorithms in Spark take as input a Vector type
    val vectorAssembler = new VectorAssembler()
    .setInputCols(Array("a", "b", "label"))
    .setOutputCol("features")

    // Perform pipeline with sequence of stages to process and learn from data
    val transformationPipeline = new Pipeline()
    .setStages(Array(vectorAssembler))
    val pipeLine = transformationPipeline.fit(data)
    val transformedTraining = pipeLine.transform(data)
    
    import scala.collection.mutable.ArrayBuffer
    val clusters = ArrayBuffer [Int] ()
    val costs = ArrayBuffer [Double] ()

    // Calculating the cost (sum of squared distances of points to their nearest center)
    for (i <- low to high) {
      
      val kmeans = new KMeans()
      .setK(i)
      .setSeed(1L)
      val kmModel = kmeans.fit(transformedTraining)
      val cost = kmModel.computeCost(transformedTraining)
      
      clusters += i
      costs += cost
      
    }
    
    val pairs = clusters.toArray.zip(costs)
    return pairs
    
  }
   
  println("Task 4:")
  val answer4 = task4(dataK5D2, 2, 10)
  answer4.foreach(println)
     
  }

