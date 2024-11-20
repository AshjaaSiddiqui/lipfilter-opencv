using System.Collections.Generic;
using UnityEngine;
using DlibFaceLandmarkDetector;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;
using UnityEngine.UI;

public class LipDetector : MonoBehaviour
{
    [SerializeField] RawImage img;  // RawImage to display webcam feed

    public string shapePredictorFilePath;
    public Color32 lipColor = Color.red; // Color to apply to the lips
    [Range(0f, 1f)] public float transparency = 0.5f; // Transparency level (0 = fully transparent, 1 = fully opaque)
    [Range(0f, 10f)] public float blurAmount = 2f; // Blur amount for the outer edges
    [Range(0, 20)] public int dilationSize = 5; // Dilation size for expanding the lip mask

    private WebCamTexture webCamTexture;
    private FaceLandmarkDetector faceLandmarkDetector;
    private Texture2D texture;

    void Start()
    {
        // Initialize the WebCamTexture
        webCamTexture = new WebCamTexture();
        img.texture = webCamTexture;

        // Start the webcam feed
        webCamTexture.Play();

        // Set the shape predictor file path
        shapePredictorFilePath = "Assets/Datfiles/shape_predictor_68_face_landmarks.dat";

        // Initialize the FaceLandmarkDetector
        faceLandmarkDetector = new FaceLandmarkDetector(shapePredictorFilePath);

        // Create a new Texture2D with the same dimensions as the webcam feed
        texture = new Texture2D(webCamTexture.width, webCamTexture.height);
    }

    void Update()
    {
        // Only process if webcam has updated the frame
        if (webCamTexture != null && webCamTexture.didUpdateThisFrame)
        {
            ProcessFrame();

            // Apply the updated texture to the RawImage after processing the frame
            img.texture = texture;
        }
    }

    private void ProcessFrame()
    {
        // Get pixel data from WebCamTexture
        Color32[] pixels = webCamTexture.GetPixels32();

        // Set the webcam image for face detection
        faceLandmarkDetector.SetImage(webCamTexture, pixels);

        // Detect faces in the image
        List<UnityEngine.Rect> faces = faceLandmarkDetector.Detect();

        // Initialize the lip color with transparency once
        Color32 transparentLipColor = lipColor;
        transparentLipColor.a = (byte)(transparency * 255);

        foreach (UnityEngine.Rect faceRect in faces)
        {
            // Detect landmarks for the face
            List<Vector2> landmarks = faceLandmarkDetector.DetectLandmark(faceRect);

            // Apply lip color and blur using the detected landmarks
            ApplyLipColorAndBlur(landmarks, pixels, transparentLipColor);
        }

        // Apply the changes to the texture
        texture.SetPixels32(pixels);
        texture.Apply();
    }

    private void ApplyLipColorAndBlur(List<Vector2> landmarks, Color32[] pixels, Color32 transparentLipColor)
    {
        // Indices of lip landmarks in a 68-point model (adjust if necessary)
        int[] outerLipIndices = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
        int[] innerLipIndices = { 60, 61, 62, 63, 64, 65, 66, 67 };

        // Create a list of vertices for the outer lip area
        List<Point> outerLip = new List<Point>(outerLipIndices.Length);
        foreach (int index in outerLipIndices)
        {
            if (index < landmarks.Count)
                outerLip.Add(new Point(landmarks[index].x, landmarks[index].y));
        }

        // Create a list of vertices for the inner lip area
        List<Point> innerLip = new List<Point>(innerLipIndices.Length);
        foreach (int index in innerLipIndices)
        {
            if (index < landmarks.Count)
                innerLip.Add(new Point(landmarks[index].x, landmarks[index].y));
        }

        // Convert pixels array to Mat (OpenCV format)
        Mat imgMat = new Mat(webCamTexture.height, webCamTexture.width, CvType.CV_8UC4);
        Utils.copyToMat(pixels, imgMat);

        // Generate a mask for the lips based on the landmarks using Convex Hull
        Mat lipMask = new Mat(imgMat.size(), CvType.CV_8UC1, new Scalar(0));  // Single channel mask

        // Create the convex hull for the outer and inner lips
        List<MatOfPoint> lipContours = new List<MatOfPoint>
    {
        new MatOfPoint(outerLip.ToArray()),
        new MatOfPoint(innerLip.ToArray())
    };

        // Draw filled polygon on the lip mask
        Imgproc.fillPoly(lipMask, lipContours, new Scalar(255));

        // Apply dilation to expand the mask
        Mat dilatedMask = new Mat();
        Imgproc.dilate(lipMask, dilatedMask, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(dilationSize, dilationSize)));

        // Apply Gaussian blur to the dilated mask for the blur effect
        Mat blurredMask = new Mat();
        Imgproc.GaussianBlur(dilatedMask, blurredMask, new Size(blurAmount * 2 + 1, blurAmount * 2 + 1), 0);

        // Flip the lip mask vertically
        Core.flip(blurredMask, blurredMask, 0);  // 0 = vertical flip

        // Use the blurred mask to apply the lip color onto the original image
        int imgCols = imgMat.cols();
        int imgRows = imgMat.rows();
        byte[] maskData = new byte[imgCols * imgRows];
        blurredMask.get(0, 0, maskData); // Get mask data in one call

        for (int i = 0; i < pixels.Length; i++)
        {
            if (maskData[i] == 255)
            {
                // Apply the lip color with transparency
                pixels[i] = Color32.Lerp(pixels[i], transparentLipColor, transparency);
            }
        }
    }

}
