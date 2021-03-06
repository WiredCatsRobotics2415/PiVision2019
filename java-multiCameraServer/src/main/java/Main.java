/*----------------------------------------------------------------------------*/
/* Copyright (c) 2018 FIRST. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.io.IOException;
import java.nio.channels.Pipe;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;

import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.RotatedRect;
import edu.wpi.cscore.*;

/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
   }
 */

public final class Main {
  private static String configFile = "/boot/frc.json";

  @SuppressWarnings("MemberName")
  public static class CameraConfig {
    public String name;
    public String path;
    public JsonObject config;
    public JsonElement streamConfig;
  }

  public static int team;
  public static boolean server;
  public static List<CameraConfig> cameraConfigs = new ArrayList<>();

  private Main() {
  }

  /**
   * Report parse error.
   */
  public static void parseError(String str) {
    System.err.println("config error in '" + configFile + "': " + str);
  }

  /**
   * Read single camera configuration.
   */
  public static boolean readCameraConfig(JsonObject config) {
    CameraConfig cam = new CameraConfig();

    // name
    JsonElement nameElement = config.get("name");
    if (nameElement == null) {
      parseError("could not read camera name");
      return false;
    }
    cam.name = nameElement.getAsString();

    // path
    JsonElement pathElement = config.get("path");
    if (pathElement == null) {
      parseError("camera '" + cam.name + "': could not read path");
      return false;
    }
    cam.path = pathElement.getAsString();

    // stream properties
    cam.streamConfig = config.get("stream");

    cam.config = config;

    cameraConfigs.add(cam);
    return true;
  }

  /**
   * Read configuration file.
   */
  @SuppressWarnings("PMD.CyclomaticComplexity")
  public static boolean readConfig() {
    // parse file
    JsonElement top;
    try {
      top = new JsonParser().parse(Files.newBufferedReader(Paths.get(configFile)));
    } catch (IOException ex) {
      System.err.println("could not open '" + configFile + "': " + ex);
      return false;
    }

    // top level must be an object
    if (!top.isJsonObject()) {
      parseError("must be JSON object");
      return false;
    }
    JsonObject obj = top.getAsJsonObject();

    // team number
    JsonElement teamElement = obj.get("team");
    if (teamElement == null) {
      parseError("could not read team number");
      return false;
    }
    team = teamElement.getAsInt();

    // ntmode (optional)
    if (obj.has("ntmode")) {
      String str = obj.get("ntmode").getAsString();
      if ("client".equalsIgnoreCase(str)) {
        server = false;
      } else if ("server".equalsIgnoreCase(str)) {
        server = true;
      } else {
        parseError("could not understand ntmode value '" + str + "'");
      }
    }

    // cameras
    JsonElement camerasElement = obj.get("cameras");
    if (camerasElement == null) {
      parseError("could not read cameras");
      return false;
    }
    JsonArray cameras = camerasElement.getAsJsonArray();
    for (JsonElement camera : cameras) {
      if (!readCameraConfig(camera.getAsJsonObject())) {
        return false;
      }
    }

    return true;
  }

  /**
   * Start running the camera.
   */
  public static UsbCamera startCamera(CameraConfig config) {
    System.out.println("Starting camera '" + config.name + "' on " + config.path);
    CameraServer inst = CameraServer.getInstance();
    UsbCamera camera = new UsbCamera(config.name, config.path);
    MjpegServer server = inst.startAutomaticCapture(camera);

    Gson gson = new GsonBuilder().create();

    camera.setConfigJson(gson.toJson(config.config));
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);

    if (config.streamConfig != null) {
      server.setConfigJson(gson.toJson(config.streamConfig));
    }

    return camera;
  }

  /**
   * Example pipeline.
   */
  public static class ReflectiveTapePipeline implements VisionPipeline {
    private int[][] hsvValues;
    private double[][] hwRatio;
    private double[][] angle;
    private double minSize;
    private CvSource cvSource;

    public ReflectiveTapePipeline() {
      this.hsvValues = new int[3][2];
      this.hwRatio = new double[2][2];
      this.angle = new double[2][2];
      this.minSize = 0.0;
      this.cvSource = CameraServer.getInstance().putVideo("Default name", 680, 480);
    }
    public ReflectiveTapePipeline(int[][] hsvValues, double[][] hwRatio, double[][] angle, double minSize, int height, int width, String name) {
      this.hsvValues = hsvValues.clone();
      this.hwRatio = hwRatio.clone();
      this.angle = angle.clone();
      this.minSize = minSize;
      this.cvSource = CameraServer.getInstance().putVideo(name, height, width);
    }
    @Override
    public void process(Mat mat) {
      Mat hsv = new Mat();
      List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
      Mat hierarchy = new Mat();
      int contourMode, contourMethod;
      List<RotatedRect> boundingBoxes = new ArrayList<RotatedRect>();
      List<MatOfPoint> contourDraw = new ArrayList<MatOfPoint>();
      Mat tempForDrawing;
      try {
        Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV);
      } catch(CvException e) {
        System.out.println(e);
      }
      try {
        Core.inRange(hsv, new Scalar(hsvValues[0][0], hsvValues[1][0], hsvValues[2][0]), new Scalar(hsvValues[0][1], hsvValues[1][1], hsvValues[2][1]), hsv);
      } catch(CvException e) {
        System.out.println(e);
      }
      contourMode = Imgproc.RETR_EXTERNAL;
      contourMethod = Imgproc.CHAIN_APPROX_SIMPLE;
      try {
        Imgproc.findContours(hsv, contours, hierarchy, contourMode, contourMethod);
      } catch(CvException e) {
        System.out.println(e);
      }
      for(int i = 0; i < contours.size(); i++) {
        boundingBoxes.add(Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray())));
      }
      for(int i = 0; i < boundingBoxes.size(); i++) {
        if(boundingBoxes.get(i).size.area() < minSize) {
          //System.out.println("failed 1");
          //System.out.println(""+boundingBoxes.get(i).size.height+" "+boundingBoxes.get(i).size.width);
          //System.out.println(""+hwRatio[1]+" "+hwRatio[0]);
          boundingBoxes.remove(i);
          i--;
          continue;
        }
        if(boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width < hwRatio[0][1] && boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width > hwRatio[0][0]) {
          //System.out.println("failed 2");
          //System.out.println(""+boundingBoxes.get(i).size.height+" "+boundingBoxes.get(i).size.width);
          //System.out.println(""+hwRatio[1]+" "+hwRatio[0]);
          if(boundingBoxes.get(i).angle < angle[0][1] && boundingBoxes.get(i).angle > angle[0][0]) {
            continue;
          }
        }
        if(boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width < hwRatio[1][1] && boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width > hwRatio[1][0]) {
          //System.out.println("failed 2");
          //System.out.println(""+boundingBoxes.get(i).size.height+" "+boundingBoxes.get(i).size.width);
          //System.out.println(""+hwRatio[1]+" "+hwRatio[0]);
          if(boundingBoxes.get(i).angle < angle[1][1] && boundingBoxes.get(i).angle > angle[1][0]) {
            continue;
          }
        }
        boundingBoxes.remove(i);
        i--;
      }
      double x,y,width,height,angle;
      double dHeightX, dHeightY, dWidthX, dWidthY;
      List<MatOfPoint> points = new ArrayList<MatOfPoint>();
      for(int i = 0; i < boundingBoxes.size(); i++) {
        x = boundingBoxes.get(i).center.x;
        y = boundingBoxes.get(i).center.y;
        width = boundingBoxes.get(i).size.width;
        height = boundingBoxes.get(i).size.height;
        angle = boundingBoxes.get(i).angle;
        dHeightX = Math.cos(angle/90.0*Math.PI)*height/2;
        dHeightY = Math.sin(angle/90.0*Math.PI)*height/2;
        dWidthX = Math.sin(angle/90.0*Math.PI)*width/2;
        dWidthY = Math.cos(angle/90.0*Math.PI)*width/2;
        points.add(new MatOfPoint(new Point(x+dHeightX-dWidthX,y+dHeightY+dWidthY),new Point(x+dHeightX+dWidthX,y+dHeightY-dWidthY),new Point(x-dHeightX+dWidthX,y-dHeightY-dWidthY),new Point(x-dHeightX-dWidthX,y-dHeightY+dWidthY)));
        Imgproc.rectangle(mat, new Point(x-(width/2),y-(height/2)), new Point(x+(width/2),y+(height/2)), new Scalar(0,255,255), 1);
      }
      System.out.println(points.size());
      Imgproc.polylines(mat, points, true, new Scalar(0,255,0));
      Imgproc.drawContours(mat, contours, 0, new Scalar(0,0,255));
      //Imgproc.cvtColor(hsv, tester, Imgproc.COLOR_2BGR);
      cvSource.putFrame(mat);
    }

    public void process(Mat mat, boolean compute) {
      if(compute) {
        process(mat);
      } else {
        cvSource.putFrame(mat);
      }
    }
    
    public void setValues(int[][] hsvValues, double[][] hwRatio, double[][] angle, double minSize) {
      this.hsvValues = hsvValues.clone();
      this.hwRatio = hwRatio.clone();
      this.angle = angle.clone();
      this.minSize = minSize;
    }

    private void sortRects(List<RotatedRect> rects, List<RotatedRect> left, List<RotatedRect> right) {
      for(int i = 0; i < rects.size(); i++) {
        if(rects.get(i).angle > angle[0][0] && rects.get(i).angle < angle[0][1]) {
          left.add(rects.get(i));
        } 
        if(rects.get(i).angle > angle[1][0] && rects.get(i).angle < angle[1][1]) {
          right.add(rects.get(i));
        }
      }
    }
  }

  public static class PipelineTuner extends Thread {
    private int[][] hsvValues;
    private double[][] hwRatio;
    private double[][] angle;
    private double minSize;
    private int exposure;
    private UsbCamera camera;
    private ReflectiveTapePipeline pipeline;
    private NetworkTableEntry hsvHMin,hsvHMax,hsvSMin,hsvSMax,hsvVMin,hsvVMax,hwRatio1Min,hwRatio1Max,hwRatio2Min,hwRatio2Max,angle1Min,angle1Max,angle2Min,angle2Max,minSizeEntry, exposureEntry, ringlight;
    private CvSink cvSink;
    private String name;

    public PipelineTuner(UsbCamera camera, ReflectiveTapePipeline pipeline, String name) {
      super(name);
      this.name = name;
      this.hsvValues = new int[3][2];
      this.hwRatio = new double[2][2];
      this.angle = new double[2][2];
      this.minSize = 0.0;
      this.exposure = 1;
      this.camera = camera;
      this.pipeline = pipeline;
      this.cvSink = CameraServer.getInstance().getVideo(camera);
      NetworkTableInstance inst = NetworkTableInstance.getDefault();
      NetworkTable table = inst.getTable(name);
      this.hsvHMin = table.getEntry("hsvHMin");
      this.hsvHMax = table.getEntry("hsvHMax");
      this.hsvSMin = table.getEntry("hsvSMin");
      this.hsvSMax = table.getEntry("hsvSMax");
      this.hsvVMin = table.getEntry("hsvVMin");
      this.hsvVMax = table.getEntry("hsvVMax");
      this.hwRatio1Min = table.getEntry("hwRatio1Min");
      this.hwRatio1Max = table.getEntry("hwRatio1Max");
      this.hwRatio2Min = table.getEntry("hwRatio2Min");
      this.hwRatio2Max = table.getEntry("hwRatio2Max");
      this.angle1Min = table.getEntry("angle1Min");
      this.angle1Max = table.getEntry("angle1Max");
      this.angle2Min = table.getEntry("angle2Min");
      this.angle2Max = table.getEntry("angle2Max");
      this.minSizeEntry = table.getEntry("minSizeEntry");
      this.exposureEntry = table.getEntry("exposureEntry");
      this.ringlight = table.getEntry("ringlight");
      hsvHMin.setDefaultDouble(hsvValues[0][0]);
      hsvHMax.setDefaultDouble(hsvValues[0][1]);
      hsvSMin.setDefaultDouble(hsvValues[1][0]);
      hsvSMax.setDefaultDouble(hsvValues[1][1]);
      hsvVMin.setDefaultDouble(hsvValues[2][0]);
      hsvVMax.setDefaultDouble(hsvValues[2][1]);
      hwRatio1Min.setDefaultDouble(hwRatio[0][0]);
      hwRatio1Max.setDefaultDouble(hwRatio[0][1]);
      hwRatio2Min.setDefaultDouble(hwRatio[1][0]);
      hwRatio2Max.setDefaultDouble(hwRatio[1][1]);
      angle1Min.setDefaultDouble(angle[0][0]);
      angle1Max.setDefaultDouble(angle[0][1]);
      angle2Min.setDefaultDouble(angle[1][0]);
      angle2Max.setDefaultDouble(angle[1][1]);
      minSizeEntry.setDefaultDouble(minSize);
      exposureEntry.setDefaultDouble(exposure);
      System.out.println("intitiated");
      inst.startClientTeam(2415);
    }
    @Override
    public void run() {
      while(true) {
      Mat img = new Mat();
      cvSink.grabFrame(img);
      hsvValues[0][0] = (int)Math.round(hsvHMin.getDouble(hsvValues[0][0]));
      hsvValues[0][1] = (int)Math.round(hsvHMax.getDouble(hsvValues[0][1]));
      hsvValues[1][0] = (int)Math.round(hsvSMin.getDouble(hsvValues[1][0]));
      hsvValues[1][1] = (int)Math.round(hsvSMax.getDouble(hsvValues[1][1]));
      hsvValues[2][0] = (int)Math.round(hsvVMin.getDouble(hsvValues[2][0]));
      hsvValues[2][1] = (int)Math.round(hsvVMax.getDouble(hsvValues[2][1]));
      hwRatio[0][0] = hwRatio1Min.getDouble(hwRatio[0][0]);
      hwRatio[0][1] = hwRatio1Max.getDouble(hwRatio[0][1]);
      hwRatio[1][0] = hwRatio2Min.getDouble(hwRatio[1][0]);
      hwRatio[1][1] = hwRatio2Max.getDouble(hwRatio[1][1]);
      angle[0][0] = angle1Min.getDouble(angle[0][0]);
      angle[0][1] = angle1Max.getDouble(angle[0][1]);
      angle[1][0] = angle2Min.getDouble(angle[1][0]);
      angle[1][1] = angle2Max.getDouble(angle[1][1]);
      minSize = minSizeEntry.getDouble(minSize);
      exposure = (int)Math.round(exposureEntry.getDouble(exposure));
      if(exposure < 0 || exposure > 100) {
        camera.setExposureAuto();
        ringlight.forceSetBoolean(false);
      } else {
        camera.setExposureManual(exposure);
        ringlight.forceSetBoolean(true);
      }
      pipeline.setValues(hsvValues, hwRatio, angle, minSize);
      if(img.width() != 0) {
        if(exposure < 0 || exposure > 100) {
          pipeline.process(img, false);
        } else {
          pipeline.process(img, true);
        }
      } else {
        System.out.println("no image");
      }
      }
    }
  }

  /**
   * Main.
   */
  public static void main(String... args) {
    if (args.length > 0) {
      configFile = args[0];
    }

    // read configuration
    if (!readConfig()) {
      return;
    }

    // start NetworkTables
    NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
    if (server) {
      System.out.println("Setting up NetworkTables server");
      ntinst.startServer();
    } else {
      System.out.println("Setting up NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
    }

    // start cameras
    List<UsbCamera> cameras = new ArrayList<>();
    for (CameraConfig cameraConfig : cameraConfigs) {
      cameras.add(startCamera(cameraConfig));
    }

    // start image processing on camera 0 if present
    if (cameras.size() >= 1) {
      PipelineTuner pipeline = new PipelineTuner(cameras.get(0), new ReflectiveTapePipeline(), "CameraHi");
      /* something like this for GRIP:
      VisionThread visionThread = new VisionThread(cameras.get(0),
              new GripPipeline(), pipeline -> {
        ...
      });
       */
      pipeline.start();
    }

    // loop forever
    for (;;) {
      try {
        Thread.sleep(10000);
      } catch (InterruptedException ex) {
        return;
      }
    }
  }
}
