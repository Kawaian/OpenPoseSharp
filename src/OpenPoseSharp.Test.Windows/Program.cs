using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace OpenPoseSharp.Test.Windows
{
    public class HandLandmarkDetector : IDisposable
    {
        public class LandmarkResult
        {
            public LandmarkPoint[] LeftHand { get; set; }
            public LandmarkPoint[] RightHand { get; set; }
        }

        public class LandmarkPoint
        {
            public double X { get; set; }
            public double Y { get; set; }
            public double Prob { get; set; }
        }

        public class Rect
        {
            public double X, Y, Width, Height;

            public Rect() { }
            public Rect(double x, double y, double w, double h)
            {
                X = x;
                Y = y;
                Width = w;
                Height = h;
            }
        }

        public int InputSize { get; set; }
        public int GPUID { get; set; }
        public string ModelDirectory { get; set; }
        public HandExtractorCaffe Extractor { get; protected set; }

        public HandLandmarkDetector(string modelDir, int netSize = 386, int gpuId = 0)
        {
            ModelDirectory = modelDir;
            GPUID = gpuId;
            InputSize = netSize;
        }

        double[] ParseText(string s)
        {
            var ret = new List<double>();
            var splSpace = s.Split(' ');
            foreach (var itemSpace in splSpace)
            {
                var spl = itemSpace.Split('\n');
                foreach (var item in spl)
                {
                    double d;
                    if (double.TryParse(item.Trim(), out d))
                    {
                        ret.Add(d);
                    }
                }
            }
            return ret.ToArray();
        }

        public void Init()
        {
            Extractor = new HandExtractorCaffe(new IntPoint(InputSize, InputSize), new IntPoint(InputSize, InputSize), ModelDirectory, GPUID);
            Extractor.initializationOnThread();
            Extractor.netInitializationOnThread();
        }

        public LandmarkResult Detect(IntPtr mat, Rect leftRect, Rect rightRect)
        {
            var e = Extractor;
            var leftR = new Rectangle((int)leftRect.X, (int)leftRect.Y, (int)leftRect.Width, (int)leftRect.Height);
            var rightR = new Rectangle((int)rightRect.X, (int)rightRect.Y, (int)rightRect.Width, (int)rightRect.Height);
            e.forwardPass(new FloatRectangle2ArrayList(new[] { new FloatRectangle2Array(new[] { leftR, rightR }) }), new SWIGTYPE_p_cv__Mat(mat, true));
            var fetch = e.getHandKeypoints();

            var left = new List<LandmarkPoint>();
            var right = new List<LandmarkPoint>();
            for (int f = 0; f < 2; f += 1)
            {
                var result = fetch[f];
                var str = result.toString();
                var parsed = ParseText(str);
                double ptX = 0, ptY = 0;
                for (int i = 0; i < parsed.Length; i++)
                {
                    switch (i % 3)
                    {
                        case 0:
                            ptX = parsed[i];
                            break;
                        case 1:
                            ptY = parsed[i];
                            break;
                        case 2:
                            var prob = parsed[i];
                            if (f == 0)
                                left.Add(new LandmarkPoint() { Prob = prob, X = ptX, Y = ptY });
                            else
                                right.Add(new LandmarkPoint() { Prob = prob, X = ptX, Y = ptY });
                            break;
                    }
                }
            }

            var ret = new LandmarkResult()
            {
                LeftHand = left.ToArray(),
                RightHand = right.ToArray()
            };
            return ret;
        }

        public void Dispose()
        {
            Extractor?.Dispose();
            Extractor = null;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"EnvPath: {Environment.CurrentDirectory}");
            var modelDir = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Environment.CurrentDirectory))))), "external", "openpose", "models") + Path.DirectorySeparatorChar;
            Console.WriteLine($"ModelPath: {modelDir}");

            using (var capture = VideoCapture.FromCamera(0))
            using (var e = new HandLandmarkDetector(modelDir, 256))
            {
                e.Init();

                int w = (int)(capture.FrameWidth * 0.75);
                var rect = new Rect(capture.FrameWidth / 2 - w / 2, capture.FrameHeight / 2 - w / 2, w, w);

                while (true)
                {
                    using (var frame = capture.RetrieveMat())
                    {
                        Cv2.Rectangle(frame, rect, Scalar.Red, 2);

                        var handrect = new HandLandmarkDetector.Rect(rect.X, rect.Y, rect.Width, rect.Height);
                        var result = e.Detect(frame.CvPtr, handrect, handrect);

                        for (int i = 0; i < 2; i++)
                        {
                            var marks = i == 0 ? result.LeftHand : result.RightHand;
                            double prePtY = 0, prePtX = 0;
                            foreach (var item in marks)
                            {
                                double prob = item.Prob, ptX = item.X, ptY = item.Y;
                                if (prob > 0.15)
                                {
                                    var probColor = new Scalar(255 * prob, 0, (1 - prob) * 255);
                                    if (prePtY != 0)
                                        Cv2.Line(frame, new Point(ptX, ptY), new Point(prePtX, prePtY), probColor, 3);
                                    Cv2.Rectangle(frame, new Rect((int)ptX - 6, (int)ptY - 6, 12, 12), Scalar.Lime, -1);
                                    Cv2.Rectangle(frame, new Rect((int)ptX - 6, (int)ptY - 6, 12, 12), probColor, 2);
                                }
                                Cv2.PutText(frame, (i / 3).ToString(), new Point(ptX, ptY + 1), HersheyFonts.HersheyPlain, 1, Scalar.Black);
                                Cv2.PutText(frame, (i / 3).ToString(), new Point(ptX, ptY), HersheyFonts.HersheyPlain, 1, Scalar.Cyan);
                                prePtX = ptX;
                                prePtY = ptY;
                            }
                        }

                        Cv2.ImShow("camera", frame);
                        Cv2.WaitKey(1);
                    }
                }
            }
        }
    }
}
