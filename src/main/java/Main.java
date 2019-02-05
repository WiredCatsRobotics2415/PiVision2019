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
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.vision.VisionThread;

import org.opencv.core.Core;
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
    private double[] hwRatio;
    private double[][] angle;
    private double minSize;
    private CvSource cvSource;

    public ReflectiveTapePipeline() {
      this.hsvValues = new int[3][2];
      this.hwRatio = new double[2];
      this.angle = new double[2][2];
      this.minSize = 0.0;
      this.cvSource = CameraServer.getInstance().putVideo(name, height, width);
    }
    public ReflectiveTapePipeline(int[][] hsvValues, double[] hwRatio, double[][] angle, double minSize, double height, double width, String name) {
      this.hsvValues = hsvValues.clone();
      this.hwRatio = hwRatio.clone();
      this.angle = angle.clone();
      this.minSize = minSize;
      this.cvSource = CameraServer.getInstance().putVideo(name, height, width);
    }
    @Override
    public void process(Mat mat) {
      Mat hsv;
      List<MatOfPoint> contours;
      Mat hierarchy;
      int contourMode, contourMethod;
      List<RotatedRect> boundingBoxes;
      Mat tempForDrawing;
      try {
        Imgproc.cvtColor(mat, hsv, Imgproc.COLOR_BGR2HSV);
      } catch(CvException e) {
        System.out.println(e);
      }
      try {
        Core.inRange(hsv, new Scalar(hsvValues[0][0], hsvValues[1][0], hsvValues[2][0]), new Scalar(hsvValues[0][0], hsvValues[1][1], hsvValues[2][1]), hsv);
      } catch(CvException e) {
        System.out.println(e);
      }
      contourMode = Imgproc.RETR_EXTERNAL;
      contourMethod = Imgproc.CHAIN_APPROX_SIMPLE;
      try {
        Imgproc.findContours(hsv, contours, hierarchy, countourMode, contourMethod);
      } catch(CvException e) {
        System.out.println(e);
      }
      contours2f = new ArrayList<MatOfPoint2f>();
      for(int i = 0; i < contours.size(); i++) {
        boundingBoxes.add(Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray())));
      }
      for(int i = 0; i < boundingBoxes.size(); i++) {
        if(boundingBoxes.get(i).size.area() < minSize) {
          boundingBoxes.remove(i);
          i--;
        }
        if(boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width < hwRatio[0] || boundingBoxes.get(i).size.height/boundingBoxes.get(i).size.width > hwRatio[1]) {
          boundingBoxes.remove(i);
          i--;
        }
        if((boundingBoxes.get(i).angle < angle[0][0] || boundingBoxes.get(i) > angle[0][1]) && (boundingBoxes.get(i).angle < angle[1][0] || boundingBoxes.get(i) > angle[1][1])) {
          boundingBoxes.remove(i);
          i--;
        }
      }
      for(int i = 0; i < boundingBoxes.size(); i++) {
        tempForDrawing = new Mat();
        Imgproc.boxPoints(tempForDrawing);
        Imgproc.drawContours(mat, new MatOfPoint(tempForDrawing), 0, new Scaler(0,0,255));
      }
      cvSource.putFrame(mat);
    }

    public void process(Mat mat, boolean compute) {
      if(compute) {
        process(mat);
      } else {
        cvSource.putFrame(mat);
      }
    }
    
    public void setValues(int[][] hsvValues, double[] hwRatio, double[][] angle, double minSize, double height, double width) {
      this.hsvValues = hsvValues.clone();
      this.hwRatio = hwRatio.clone();
      this.angle = angle.clone();
    }
  }

  public static class PipelineTuner extends Thread {
    private int[][] hsvValues;
    private double[] hwRatio;
    private double[][] angle;
    private double minSize;
    private int exposure;
    private UsbCamera camera;
    private ReflectiveTapePipeline pipeline;
    private NetworkTableEntry hsvHMin,hsvHMax,hsvSMin,hsvSMax,hsvVMin,hsvVMax,hwRatioMin,hwRatioMax,angle1Min,angle1Max,angle2Min,angle2Max,minSizeEntry, exposureEntry, ringlight;
    private CvSink cvSink;
    private String name;

    public PipelineTuner(UsbCamera camera, ReflectiveTapePipeline pipeline, String name) {
      super(name);
      this.name = name;
      this.hsvValues = new int[3][2];
      this.hwRatio = new double[2];
      this.angle = new double[2][2];
      this.minSize = 0.0;
      this.exposure = -1;
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
      this.hwRatioMin = table.getEntry("hwRatioMin");
      this.hwRatioMax = table.getEntry("hwRatioMax");
      this.angle1Min = table.getEntry("angle1Min");
      this.angle1Max = table.getEntry("angle1Max");
      this.angle2Min = table.getEntry("angle2Min");
      this.angle2Max = table.getEntry("angle2Max");
      this.minSizeEntry = table.getEntry("minSizeEntry");
      this.exposureEntry = table.getEntry("exposureEntry");
      this.ringlight = table.getEntry("ringlight");
      inst.startClientTeam(2415);
    }
    @Override
    public void run() {
      Mat img = new Mat();
      cvSink.grabFrame(img);
      hsvValues[0][0] = (int)Math.round(hsvHMin.getDouble(hsvValues[0][0]));
      hsvValues[0][1] = (int)Math.round(hsvHMax.getDouble(hsvValues[0][1]));
      hsvValues[1][0] = (int)Math.round(hsvSMin.getDouble(hsvValues[1][0]));
      hsvValues[1][1] = (int)Math.round(hsvSMax.getDouble(hsvValues[1][1]));
      hsvValues[2][0] = (int)Math.round(hsvVMin.getDouble(hsvValues[2][0]));
      hsvValues[2][1] = (int)Math.round(hwRatioMin.getDouble(hsvValues[2][1]));
      hwRatio[0] = hwRatioMin.getDouble(hwRatio[0]);
      hwRatio[1] = hwRatioMin.getDouble(hwRatio[1]);
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
      pipeline.setValues(hsvValues, hwRatio, angle, minSize, height, width);
      if(img.width() != 0) {
        if(exposure < 0 || exposure > 100) {
          pipeline.process(img, false);
        } else {
          pipeline.process(img, true);
        }
      } else {
        System.out.println("No image");
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
      PipelineTuner pipeline = new PipelineTuner(cameras.get(0), new ReflectiveTapePipeline(), "Camera0");
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
