#pragma warning disable 0219
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

#if DLIB_USE_UNSAFE_CODE && UNITY_2018_2_OR_NEWER
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;
#endif

namespace DlibFaceLandmarkDetector
{
    /// <summary>
    /// Face landmark detector.
    /// </summary>
    public class FaceLandmarkDetector : DisposableDlibObject
    {
        public static FaceLandmarkDetector Instance { get; private set; }

        public FaceLandmarkDetector()
        {
            if (Instance != null)
            {
                throw new InvalidOperationException("There can only be one instance of FaceLandmarkDetector.");
            }
            Instance = this;
        }

        protected override void Dispose(bool disposing)
        {
            try
            {
                if (disposing)
                {
                }

                if (IsEnabledDispose)
                {
                    if (nativeObj != IntPtr.Zero)
                        DlibFaceLandmarkDetector_Dispose(nativeObj);
                    nativeObj = IntPtr.Zero;
                }
            }
            finally
            {
                base.Dispose(disposing);
            }
        }

        /**
        * Initializes a new instance of the FaceLandmarkDetector class.
        * <p>
        * <br>This instance uses default frontal face detector.
        * 
        * ObjectDetector is initialized in such a code.
        *   frontal_face_detector face_detector;
        *   face_detector = get_frontal_face_detector();
        * 
        * ShapePredictor is initialized in such a code.
        *   shape_predictor sp;
        *   deserialize(shape_predictor_filename) >> sp;
        * 
        * @param shapePredictorFilePath
        */
        public FaceLandmarkDetector(string shapePredictorFilePath)
        {
            nativeObj = DlibFaceLandmarkDetector_Init();

            if (!DlibFaceLandmarkDetector_LoadObjectDetector(nativeObj, null))
            {
                //Debug.LogError ("Failed to load " + objectDetectorFilename);
            }

            if (!DlibFaceLandmarkDetector_LoadShapePredictor(nativeObj, shapePredictorFilePath))
            {
                Debug.LogError("Failed to load " + shapePredictorFilePath);
            }
        }

        /**
        * Initializes a new instance of the FaceLandmarkDetector class.
        * 
        * ObjectDetector is initialized in such a code.
        *   if(object_detector_filename != null){
        *     object_detector<scan_fhog_pyramid<pyramid_down<6>>> simple_detector;
        *     deserialize(object_detector_filename) >> simple_detector;
        *   }else{
        *     frontal_face_detector face_detector;
        *     face_detector = get_frontal_face_detector();
        *   }
        * 
        * ShapePredictor is initialized in such a code.
        *   shape_predictor sp;
        *   deserialize(shape_predictor_filename) >> sp;
        * 
        * @param objectDetectorFilePath
        * @param shapePredictorFilePath
        */
        public FaceLandmarkDetector(string objectDetectorFilePath, string shapePredictorFilePath)
        {
            nativeObj = DlibFaceLandmarkDetector_Init();

            if (!DlibFaceLandmarkDetector_LoadObjectDetector(nativeObj, objectDetectorFilePath))
            {
                Debug.LogError("Failed to load " + objectDetectorFilePath);
            }

            if (!DlibFaceLandmarkDetector_LoadShapePredictor(nativeObj, shapePredictorFilePath))
            {
                Debug.LogError("Failed to load " + shapePredictorFilePath);
            }
        }

        /**
        * Sets Image from Texture2D.
        * 
        * @param texture2D
        */
        public void SetImage(Texture2D texture2D)
        {
            if (texture2D == null)
                throw new ArgumentNullException("texture2D");
            ThrowIfDisposed();

            if (texture2D.format == TextureFormat.RGBA32 || texture2D.format == TextureFormat.RGB24 || texture2D.format == TextureFormat.Alpha8)
            {
#if DLIB_USE_UNSAFE_CODE && UNITY_2018_2_OR_NEWER
                unsafe
                {
                    SetImage ((IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(texture2D.GetRawTextureData<byte>()), texture2D.width, texture2D.height, (int)texture2D.format, true);
                }
#else
                SetImage<byte>(texture2D.GetRawTextureData(), texture2D.width, texture2D.height, (int)texture2D.format, true);
#endif
                return;
            }

            Color32[] colors = texture2D.GetPixels32();

            GCHandle colorsHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
            DlibFaceLandmarkDetector_SetImage(nativeObj, colorsHandle.AddrOfPinnedObject(), texture2D.width, texture2D.height, 4, true);
            colorsHandle.Free();
        }

#if !DLIB_DONT_USE_WEBCAMTEXTURE_API
#if !(PLATFORM_LUMIN && !UNITY_EDITOR)

        /**
        * Sets Image from WebCamTexture.
        * 
        * @param webCamTexture 
        */
        public void SetImage(WebCamTexture webCamTexture)
        {
            SetImage(webCamTexture, null);
        }

        /**
        * Sets Image from WebCamTexture.
        * 
        * @param webCamTexture 
        * @param bufferColors the optional array to receive pixel data.
        * You can optionally pass in an array of Color32s to use in colors to avoid allocating new memory each frame.
        * The array needs to be initialized to a length matching width * height of the texture.(<a href="http://docs.unity3d.com/ScriptReference/WebCamTexture.GetPixels32.html">http://docs.unity3d.com/ScriptReference/WebCamTexture.GetPixels32.html</a>)
        */
        public void SetImage(WebCamTexture webCamTexture, Color32[] bufferColors)
        {
            if (webCamTexture == null)
                throw new ArgumentNullException("webCamTexture");
            ThrowIfDisposed();

            GCHandle colorsHandle;
            if (bufferColors == null)
            {
                Color32[] colors = webCamTexture.GetPixels32();

                colorsHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
            }
            else
            {
                webCamTexture.GetPixels32(bufferColors);

                colorsHandle = GCHandle.Alloc(bufferColors, GCHandleType.Pinned);
            }

            DlibFaceLandmarkDetector_SetImage(nativeObj, colorsHandle.AddrOfPinnedObject(), webCamTexture.width, webCamTexture.height, 4, true);
            colorsHandle.Free();
        }

#endif
#endif

        /**
        * Sets Image from IntPtr.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        */
        public void SetImage(IntPtr intPtr, int width, int height, int bytesPerPixel)
        {
            SetImage(intPtr, width, height, bytesPerPixel, false);
        }

        /**
        * Sets Image from IntPtr.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param flip flip vertical
        */
        public void SetImage(IntPtr intPtr, int width, int height, int bytesPerPixel, bool flip)
        {
            if (intPtr == IntPtr.Zero)
                throw new ArgumentException("intPtr == IntPtr.Zero");
            ThrowIfDisposed();

            DlibFaceLandmarkDetector_SetImage(nativeObj, intPtr, width, height, bytesPerPixel, flip);
        }

        /**
        * Sets Image from Pixel Data Array.
        * 
        * @param array
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        */
        public void SetImage<T>(T[] array, int width, int height, int bytesPerPixel) where T : struct
        {
            SetImage<T>(array, width, height, bytesPerPixel, false);
        }

        /**
        * Sets Image from Pixel Data Array.
        * 
        * @param array
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param flip flip vertical
        */
        public void SetImage<T>(T[] array, int width, int height, int bytesPerPixel, bool flip) where T : struct
        {
            if (array == null)
                throw new ArgumentNullException("array");
            ThrowIfDisposed();

            GCHandle arrayHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
            DlibFaceLandmarkDetector_SetImage(nativeObj, arrayHandle.AddrOfPinnedObject(), width, height, bytesPerPixel, flip);
            arrayHandle.Free();
        }

        public class RectDetection
        {
            public Rect rect;
            public double detection_confidence;
            public long weight_index;

            public RectDetection()
            {
                rect = new Rect();
                detection_confidence = 0.0;
                weight_index = 0;
            }
        }

        /**
        * Detects Objects.
        * 
        * @return List<Rect> detected list of object's rect.
        */
        public List<Rect> Detect()
        {
            return Detect(0.0);
        }

        /**
        * Detects Objects.
        * 
        * @param adjust_threshold
        * @return List<Rect> detected list of object's rect.
        */
        public List<Rect> Detect(double adjust_threshold)
        {
            ThrowIfDisposed();

            List<Rect> rects = new List<Rect>();

            int detectCount = DlibFaceLandmarkDetector_Detect(nativeObj, adjust_threshold);
            if (detectCount > 0)
            {
                double[] result = new double[detectCount * 6];
                DlibFaceLandmarkDetector_GetDetectResult(nativeObj, result);

                for (int i = 0; i < detectCount; i++)
                {
                    rects.Add(new Rect((float)result[i * 6 + 0], (float)result[i * 6 + 1], (float)result[i * 6 + 2], (float)result[i * 6 + 3]));
                }
            }

            return rects;
        }

        /**
        * Detects Objects.
        * 
        * @return List<RectDetection> detected list of object's RectDetection.
        */
        public List<RectDetection> DetectRectDetection()
        {
            return DetectRectDetection(0.0);
        }

        /**
        * Detects Objects.
        * 
        * @param adjust_threshold
        * @return List<RectDetection> detected list of object's RectDetection.
        */
        public List<RectDetection> DetectRectDetection(double adjust_threshold)
        {
            ThrowIfDisposed();

            List<RectDetection> rectDetections = new List<RectDetection>();

            int detectCount = DlibFaceLandmarkDetector_Detect(nativeObj, adjust_threshold);
            if (detectCount > 0)
            {
                double[] result = new double[detectCount * 6];
                DlibFaceLandmarkDetector_GetDetectResult(nativeObj, result);

                for (int i = 0; i < detectCount; i++)
                {
                    RectDetection rectDetection = new RectDetection();
                    rectDetection.rect = new Rect((float)result[i * 6 + 0], (float)result[i * 6 + 1], (float)result[i * 6 + 2], (float)result[i * 6 + 3]);
                    rectDetection.detection_confidence = result[i * 6 + 4];
                    rectDetection.weight_index = (long)result[i * 6 + 5];

                    rectDetections.Add(rectDetection);
                }
            }

            return rectDetections;
        }

        /**
        * Detects Objects.
        * 
        * @return double[] detected object's data.[left_0, top_0, width_0, height_0, detection_confidence_0, weight_index_0, left_1, top_1, width_1, height_1, detection_confidence_1, weight_index_1, ...]
        */
        public double[] DetectArray()
        {
            return DetectArray(0.0);
        }

        /**
        * Detects Objects.
        * 
        * @param adjust_threshold
        * @return double[] detected object's data.[left_0, top_0, width_0, height_0, detection_confidence_0, weight_index_0, left_1, top_1, width_1, height_1, detection_confidence_1, weight_index_1, ...]
        */
        public double[] DetectArray(double adjust_threshold)
        {
            ThrowIfDisposed();

            int detectCount = DlibFaceLandmarkDetector_Detect(nativeObj, adjust_threshold);
            if (detectCount > 0)
            {
                double[] result = new double[detectCount * 6];
                DlibFaceLandmarkDetector_GetDetectResult(nativeObj, result);

                return result;
            }

            return new double[0];
        }

        /**
        * Detects Object Landmark.
        * 
        * @param left
        * @param top
        * @param width
        * @param height
        * @return List<Vector2> detected Vector2 list of object landmark.
        */
        public List<Vector2> DetectLandmark(double left, double top, double width, double height)
        {
            ThrowIfDisposed();

            List<Vector2> points = new List<Vector2>();

            int detectCount = DlibFaceLandmarkDetector_DetectLandmark(nativeObj, left, top, width, height);
            if (detectCount > 0)
            {
                double[] result = new double[detectCount * 2];
                DlibFaceLandmarkDetector_GetDetectLandmarkResult(nativeObj, result);

                for (int i = 0; i < detectCount; i++)
                {
                    points.Add(new Vector2((float)result[i * 2 + 0], (float)result[i * 2 + 1]));
                }
            }

            return points;
        }

        /**
        * Detects Object Landmark.
        * 
        * @param rect
        * @return List<Vector2> detected Vector2 list of object landmark.
        */
        public List<Vector2> DetectLandmark(Rect rect)
        {
            return DetectLandmark(rect.xMin, rect.yMin, rect.width, rect.height);
        }

        /**
        * Detects Object Landmark.
        * 
        * @param left
        * @param top
        * @param width
        * @param height
        * @return double[] detected object landmark data.[x_0, y_0, x_1, y_1, ...]
        */
        public double[] DetectLandmarkArray(double left, double top, double width, double height)
        {
            ThrowIfDisposed();

            int detectCount = DlibFaceLandmarkDetector_DetectLandmark(nativeObj, left, top, width, height);
            if (detectCount > 0)
            {
                double[] result = new double[detectCount * 2];
                DlibFaceLandmarkDetector_GetDetectLandmarkResult(nativeObj, result);

                return result;
            }

            return new double[0];
        }

        /**
        * Detects Object Landmark.
        * 
        * @param rect
        * @return double[] detected object landmark data.[x_0, y_0, x_1, y_1, ...]
        */
        public double[] DetectLandmarkArray(Rect rect)
        {
            return DetectLandmarkArray(rect.xMin, rect.yMin, rect.width, rect.height);
        }

        /**
        * Whether all of the object parts point is contained in the object rectangle?
        * 
        * @return bool
        */
        public bool IsAllPartsInRect()
        {
            ThrowIfDisposed();

            bool flag = DlibFaceLandmarkDetector_IsAllPartsInRect(nativeObj);

            return flag;
        }

        /**
        * Gets ShapePredictorNumParts.
        * 
        * @return long
        */
        public long GetShapePredictorNumParts()
        {
            ThrowIfDisposed();

            long numParts = DlibFaceLandmarkDetector_ShapePredictorNumParts(nativeObj);

            return numParts;
        }

        /**
        * Gets ShapePredictorNumFeatures.
        * 
        * @return long
        */
        public long GetShapePredictorNumFeatures()
        {
            ThrowIfDisposed();

            long numFeatures = DlibFaceLandmarkDetector_ShapePredictorNumFeatures(nativeObj);

            return numFeatures;
        }

        /**
        * Draws Detect Result.
        * 
        * @param texture2D
        * @param r
        * @param g
        * @param b
        * @param a
        * @param thickness
        */
        public void DrawDetectResult(Texture2D texture2D, int r, int g, int b, int a, int thickness, bool updateMipmaps = false, bool makeNoLongerReadable = false)
        {
            if (texture2D == null)
                throw new ArgumentNullException("texture2D");
            ThrowIfDisposed();

            if (texture2D.format == TextureFormat.RGBA32 || texture2D.format == TextureFormat.RGB24 || texture2D.format == TextureFormat.Alpha8)
            {

#if DLIB_USE_UNSAFE_CODE && UNITY_2018_2_OR_NEWER
                unsafe
                {
                    NativeArray<byte> colors = texture2D.GetRawTextureData<byte>();

                    DlibFaceLandmarkDetector_DrawDetectResult(nativeObj, (IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(colors), texture2D.width, texture2D.height, (int)texture2D.format, true, r, g, b, a, thickness);

                    texture2D.LoadRawTextureData(colors);
                    texture2D.Apply(updateMipmaps, makeNoLongerReadable);
                }
#else
                byte[] colors = texture2D.GetRawTextureData();

                GCHandle arrayHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
                DlibFaceLandmarkDetector_DrawDetectResult(nativeObj, arrayHandle.AddrOfPinnedObject(), texture2D.width, texture2D.height, (int)texture2D.format, true, r, g, b, a, thickness);
                arrayHandle.Free();

                texture2D.LoadRawTextureData(colors);
                texture2D.Apply(updateMipmaps, makeNoLongerReadable);
#endif

            }
            else
            {
                Color32[] colors = texture2D.GetPixels32();

                GCHandle arrayHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
                DlibFaceLandmarkDetector_DrawDetectResult(nativeObj, arrayHandle.AddrOfPinnedObject(), texture2D.width, texture2D.height, 4, true, r, g, b, a, thickness);
                arrayHandle.Free();

                texture2D.SetPixels32(colors);
                texture2D.Apply(updateMipmaps, makeNoLongerReadable);
            }
        }

        /**
        * Draws Detect Result.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @bytePerPixel 1 , 3 or 4
        * @param r
        * @param g
        * @param b
        * @param a
        * @param thickness
        */
        public void DrawDetectResult(IntPtr intPtr, int width, int height, int bytesPerPixel, int r, int g, int b, int a, int thickness)
        {
            DrawDetectResult(intPtr, width, height, bytesPerPixel, false, r, g, b, a, thickness);
        }

        /**
        * Draws Detect Result.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param flip flip vertical
        * @param r
        * @param g
        * @param b
        * @param a
        * @param thickness
        */
        public void DrawDetectResult(IntPtr intPtr, int width, int height, int bytesPerPixel, bool flip, int r, int g, int b, int a, int thickness)
        {
            if (intPtr == IntPtr.Zero)
                throw new ArgumentException("intPtr == IntPtr.Zero");
            ThrowIfDisposed();

            DlibFaceLandmarkDetector_DrawDetectResult(nativeObj, intPtr, width, height, bytesPerPixel, flip, r, g, b, a, thickness);
        }

        /**
        * Draws Detect Result.
        * 
        * @param array
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param r
        * @param g
        * @param b
        * @param a
        * @param thickness
        */
        public void DrawDetectResult<T>(T[] array, int width, int height, int bytesPerPixel, int r, int g, int b, int a, int thickness) where T : struct
        {
            DrawDetectResult<T>(array, width, height, bytesPerPixel, false, r, g, b, a, thickness);
        }

        /**
        * Draws Detect Result.
        * 
        * @param array
        * @param width
        * @param height
        * @param bytePerPixel 1 , 3 or 4
        * @param flip flip vertical
        * @param r
        * @param g
        * @param b
        * @param a
        * @param thickness
        */
        public void DrawDetectResult<T>(T[] array, int width, int height, int bytesPerPixel, bool flip, int r, int g, int b, int a, int thickness) where T : struct
        {
            if (array == null)
                throw new ArgumentNullException("array");
            ThrowIfDisposed();

            GCHandle arrayHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
            DlibFaceLandmarkDetector_DrawDetectResult(nativeObj, arrayHandle.AddrOfPinnedObject(), width, height, bytesPerPixel, flip, r, g, b, a, thickness);
            arrayHandle.Free();
        }

        /**
        * Draws Detect Landmark Result.
        * 
        * @param texture2D
        * @param r
        * @param g
        * @param b
        * @param a
        */
        public void DrawDetectLandmarkResult(Texture2D texture2D, int r, int g, int b, int a, bool updateMipmaps = false, bool makeNoLongerReadable = false)
        {
            if (texture2D == null)
                throw new ArgumentNullException("texture2D");
            ThrowIfDisposed();

            if (texture2D.format == TextureFormat.RGBA32 || texture2D.format == TextureFormat.RGB24 || texture2D.format == TextureFormat.Alpha8)
            {

#if DLIB_USE_UNSAFE_CODE && UNITY_2018_2_OR_NEWER
                unsafe
                {
                    NativeArray<byte> colors = texture2D.GetRawTextureData<byte>();

                    DlibFaceLandmarkDetector_DrawDetectLandmarkResult(nativeObj, (IntPtr)NativeArrayUnsafeUtility.GetUnsafeReadOnlyPtr(colors), texture2D.width, texture2D.height, (int)texture2D.format, true, r, g, b, a);

                    texture2D.LoadRawTextureData(colors);
                    texture2D.Apply(updateMipmaps, makeNoLongerReadable);
                }
#else
                byte[] colors = texture2D.GetRawTextureData();

                GCHandle arrayHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
                DlibFaceLandmarkDetector_DrawDetectLandmarkResult(nativeObj, arrayHandle.AddrOfPinnedObject(), texture2D.width, texture2D.height, (int)texture2D.format, true, r, g, b, a);
                arrayHandle.Free();

                texture2D.LoadRawTextureData(colors);
                texture2D.Apply(updateMipmaps, makeNoLongerReadable);
#endif

            }
            else
            {
                Color32[] colors = texture2D.GetPixels32();

                GCHandle arrayHandle = GCHandle.Alloc(colors, GCHandleType.Pinned);
                DlibFaceLandmarkDetector_DrawDetectLandmarkResult(nativeObj, arrayHandle.AddrOfPinnedObject(), texture2D.width, texture2D.height, 4, true, r, g, b, a);
                arrayHandle.Free();

                texture2D.SetPixels32(colors);
                texture2D.Apply(updateMipmaps, makeNoLongerReadable);
            }
        }

        /**
        * Draws Detect Landmark Result.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @param bytesPerPixel 1 ,3 or 4
        * @param r
        * @param g
        * @param b
        * @param a
        */
        public void DrawDetectLandmarkResult(IntPtr intPtr, int width, int height, int bytesPerPixel, int r, int g, int b, int a)
        {
            DrawDetectLandmarkResult(intPtr, width, height, bytesPerPixel, false, r, g, b, a);
        }

        /**
        * Draws Detect Landmark Result.
        * 
        * @param intPtr
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param flip flip vertical
        * @param r
        * @param g
        * @param b
        * @param a
        */
        public void DrawDetectLandmarkResult(IntPtr intPtr, int width, int height, int bytesPerPixel, bool flip, int r, int g, int b, int a)
        {
            if (intPtr == IntPtr.Zero)
                throw new ArgumentException("intPtr == IntPtr.Zero");
            ThrowIfDisposed();

            DlibFaceLandmarkDetector_DrawDetectLandmarkResult(nativeObj, intPtr, width, height, bytesPerPixel, flip, r, g, b, a);
        }

        /**
        * Draws Detect Landmark Result.
        * 
        * @param array
        * @param width
        * @param height
        * @param bytesPerPixel 1 , 3 or 4
        * @param r
        * @param g
        * @param b
        * @param a
        */
        public void DrawDetectLandmarkResult<T>(T[] array, int width, int height, int bytesPerPixel, int r, int g, int b, int a) where T : struct
        {
            DrawDetectLandmarkResult<T>(array, width, height, bytesPerPixel, false, r, g, b, a);
        }

        /**
        * Draws Detect Landmark Result.
        * 
        * @param array
        * @param width
        * @param height
        * @bytesPerPixel 1 , 3 or 4
        * @flip flip vertical
        * @param r
        * @param g
        * @param b
        * @param a
        */
        public void DrawDetectLandmarkResult<T>(T[] array, int width, int height, int bytesPerPixel, bool flip, int r, int g, int b, int a) where T : struct
        {
            if (array == null)
                throw new ArgumentNullException("array");
            ThrowIfDisposed();

            GCHandle arrayHandle = GCHandle.Alloc(array, GCHandleType.Pinned);
            DlibFaceLandmarkDetector_DrawDetectLandmarkResult(nativeObj, arrayHandle.AddrOfPinnedObject(), width, height, bytesPerPixel, flip, r, g, b, a);
            arrayHandle.Free();
        }


#if (UNITY_IOS || UNITY_WEBGL) && !UNITY_EDITOR
        const string LIBNAME = "__Internal";
#else
        const string LIBNAME = "dlibfacelandmarkdetector";
#endif

        [DllImport(LIBNAME)]
        private static extern IntPtr DlibFaceLandmarkDetector_Init();

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_Dispose(IntPtr nativeObj);


        [DllImport(LIBNAME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool DlibFaceLandmarkDetector_LoadObjectDetector(IntPtr self, [MarshalAs(UnmanagedType.LPWStr)] string objectDetectorFilename);

        [DllImport(LIBNAME, CharSet = CharSet.Unicode)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool DlibFaceLandmarkDetector_LoadShapePredictor(IntPtr self, [MarshalAs(UnmanagedType.LPWStr)] string shapePredictorFilename);

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_SetImage(IntPtr self, IntPtr byteArray, int texWidth, int texHeight, int bytesPerPixel, [MarshalAs(UnmanagedType.U1)] bool flip);

        [DllImport(LIBNAME)]
        private static extern int DlibFaceLandmarkDetector_Detect(IntPtr self, double adjust_threshold);

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_GetDetectResult(IntPtr self, double[] result);

        [DllImport(LIBNAME)]
        private static extern int DlibFaceLandmarkDetector_DetectLandmark(IntPtr self, double left, double top, double width, double height);

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_GetDetectLandmarkResult(IntPtr self, double[] result);

        [DllImport(LIBNAME)]
        [return: MarshalAs(UnmanagedType.U1)]
        private static extern bool DlibFaceLandmarkDetector_IsAllPartsInRect(IntPtr self);

        [DllImport(LIBNAME)]
        private static extern long DlibFaceLandmarkDetector_ShapePredictorNumParts(IntPtr self);

        [DllImport(LIBNAME)]
        private static extern long DlibFaceLandmarkDetector_ShapePredictorNumFeatures(IntPtr self);

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_DrawDetectResult(IntPtr self, IntPtr byteArray, int texWidth, int texHeight, int bytesPerPixel, [MarshalAs(UnmanagedType.U1)] bool flip, int r, int g, int b, int a, int thickness);

        [DllImport(LIBNAME)]
        private static extern void DlibFaceLandmarkDetector_DrawDetectLandmarkResult(IntPtr self, IntPtr byteArray, int texWidth, int texHeight, int bytesPerPixel, [MarshalAs(UnmanagedType.U1)] bool flip, int r, int g, int b, int a);
    }
}
#pragma warning restore 0219