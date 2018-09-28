using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace OpenPoseSharp.Test.Windows
{
    class Program
    {
        static double[] ParseText(string s)
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

        static void Main(string[] args)
        {
            Console.WriteLine($"EnvPath: {Environment.CurrentDirectory}");
            var modelDir = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Environment.CurrentDirectory))))), "external", "openpose", "models") + Path.DirectorySeparatorChar;
            Console.WriteLine($"ModelPath: {modelDir}");

            int netSize = 256;
            using (var capture = VideoCapture.FromCamera(0))
            using (var a = new HandExtractorCaffe(new IntPoint(netSize, netSize), new IntPoint(netSize, netSize), modelDir, 0))
            {
                a.initializationOnThread();
                a.netInitializationOnThread();
                var enable = a.getEnabled();

                int w = (int)(capture.FrameWidth * 0.75);
                var rect = new Rect(capture.FrameWidth / 2 - w / 2, capture.FrameHeight / 2 - w / 2, w, w);

                while (true)
                {
                    using (var frame = capture.RetrieveMat())
                    using (var roi = new Mat(frame, rect))
                    {
                        Cv2.Rectangle(frame, rect, Scalar.Red, 2);
                        a.forwardPass(new FloatRectangle2ArrayList(new[] { new FloatRectangle2Array(new[] { new Rectangle(rect.X, rect.Y, rect.Width, rect.Height), new Rectangle(rect.X, rect.Y, rect.Width, rect.Height) }) }), new SWIGTYPE_p_cv__Mat(frame.CvPtr, true));
                        var fetch = a.getHandKeypoints();
                        for (int f = 0; f < fetch.Count; f++)
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
                                        if (prob > 0.3)
                                        {
                                            Cv2.Rectangle(frame, new Rect((int)ptX, (int)ptY, 12, 12), Scalar.Lime, -1);
                                            Cv2.Rectangle(frame, new Rect((int)ptX, (int)ptY, 12, 12), new Scalar(255 * prob, 0, (1 - prob) * 255), 2);
                                        }
                                        Cv2.PutText(frame, (i / 3).ToString(), new Point(ptX, ptY), HersheyFonts.HersheyPlain, 1, Scalar.Cyan);
                                        break;
                                }
                            }
                        }
                        Cv2.ImShow("camera", frame);
                        Cv2.ImShow("roi", roi);
                        Cv2.WaitKey(1);
                    }
                }
            }

            Console.Write("Test FIN.");
            Console.ReadLine();
        }
    }
}
