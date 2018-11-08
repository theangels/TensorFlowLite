package org.blackwalnutlabs.angel.tensorflowlite.setting;

public class TensorFlowSetting {
    public static final int DIM_BATCH_SIZE = 1;
    public static final int DIM_PIXEL_SIZE = 3;
    public static final int DIM_IMG_SIZE_X = 128;
    public static final int DIM_IMG_SIZE_Y = 128;
    public static final int IMAGE_MEAN = 128;
    public static final float IMAGE_STD = 128.0f;
    public static final int RESULTS_TO_SHOW = 3;
    public static final float THRESHOLD = 0.1f;
    public static final String MODELFILE = "converted_model.tflite";
    public static final String LABEL_PATH = "retrained_labels.txt";
}
