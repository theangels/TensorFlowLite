package org.blackwalnutlabs.angel.tensorflowlite.model;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.DIM_BATCH_SIZE;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.DIM_IMG_SIZE_X;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.DIM_IMG_SIZE_Y;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.DIM_PIXEL_SIZE;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.IMAGE_MEAN;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.IMAGE_STD;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.LABEL_PATH;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.MODELFILE;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.RESULTS_TO_SHOW;
import static org.blackwalnutlabs.angel.tensorflowlite.setting.TensorFlowSetting.THRESHOLD;

public class TensorFlowLiteDetector {
    private static final String TAG = "Detector";

    // 存储图像 RGB 值
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
    // TFLite 执行类
    private Interpreter tfLite;
    // 存储标签列表
    private List<String> labelList;
    // 存储图像位数据
    private ByteBuffer imgData;
    // 存储预测结果
    private float[][] labelProbArray;

    // 初始化 TFLite
    public TensorFlowLiteDetector(Map<String, Object> othersMap) {
        try {
            Activity activity = (Activity) othersMap.get("activity");
            tfLite = new Interpreter(loadModelFile(activity));
            labelList = loadLabelList(activity);
            imgData = ByteBuffer.allocateDirect(
                    4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            imgData.order(ByteOrder.nativeOrder());
            labelProbArray = new float[1][labelList.size()];
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // 读取标签文件存储数组
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(
                        activity.getAssets().open(LABEL_PATH)
                ));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // 读取模型文件
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODELFILE);
        FileInputStream inputStream = new FileInputStream(
                fileDescriptor.getFileDescriptor()
        );
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // 格式化传入图像像素值
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(),
                0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
    }

    // 资源释放
    public void close() {
        tfLite.close();
        tfLite = null;
    }

    // 执行预测
    public List<Recognition> detectImage(Mat src) {
        if (tfLite != null) {
            Bitmap bmp = Bitmap.createBitmap(src.width(), src.height(),
                    Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(src, bmp);
            convertBitmapToByteBuffer(bmp);
            tfLite.run(imgData, labelProbArray);

            PriorityQueue<Recognition> pq =
                    new PriorityQueue<>(
                            RESULTS_TO_SHOW,
                            (lhs, rhs) -> {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            });
            for (int i = 0; i < labelProbArray[0].length; ++i) {
                if (labelProbArray[0][i] > THRESHOLD) {
                    pq.add(
                            new Recognition(
                                    "" + i, labelList.size() > i ? labelList.get(i) : "unknown", labelProbArray[0][i]));
                }
            }
            final ArrayList<Recognition> recognitions = new ArrayList<>();
            int recognitionsSize = Math.min(pq.size(), RESULTS_TO_SHOW);
            for (int i = 0; i < recognitionsSize; ++i) {
                recognitions.add(pq.poll());
            }

            return recognitions;
        }
        return null;
    }

    // 排序用
    public class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * Display name for the recognition.
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;


        Recognition(
                final String id, final String title, final Float confidence) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        Float getConfidence() {
            return confidence;
        }

        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format(Locale.getDefault(), "(%.1f%%) ", confidence * 100.0f);
            }

            return resultString.trim();
        }
    }
}
