using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

using Emgu;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.Util;

namespace A_PACS_Auto.Classes
{
    internal class ScanNum : DisposableObject
    {
        private Tesseract OCR;

        /// <summary>
        /// Экземпляр класса Tessaract
        /// </summary>
        /// <param name="pathTess">Путь к языковому классу</param>
        /// <param name="lang">Язык распознавания</param>
        public ScanNum (string pathTess, string lang)
        {
            OCR = new Tesseract(pathTess, lang, OcrEngineMode.TesseractLstmCombined);
        }

        /// <summary>
        /// Получение списка номеров
        /// </summary>
        /// <param name="image"></param>
        /// <param name="licensePlateImageList"></param>
        /// <param name="filteredePlateImageList"></param>
        /// <param name="detectedPlateImageList"></param>
        /// <returns></returns>
        public List<string> DetectNum(IInputArray image,
            List<IInputOutputArray> licensePlateImageList,
            List<IInputOutputArray> filteredePlateImageList,
            List<RotatedRect> detectedPlateImageList)
        {
            List<string> licenses = new List<string>();

            using (Mat gray = new Mat())
            {
                using (Mat canny = new Mat())
                {
                    using (VectorOfVectorOfPoint countours = new VectorOfVectorOfPoint())
                    {
                        CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);
                        CvInvoke.Canny(gray, canny, 100, 50, 3, false);
                        int[,] hierarchy = CvInvoke.FindContourTree(canny, countours, ChainApproxMethod.ChainApproxSimple);

                        FindNum(countours, hierarchy, 0, gray, canny, licensePlateImageList, filteredePlateImageList, detectedPlateImageList, licenses);
                    }
                }
            }
            return licenses;
        }

        /// <summary>
        /// Обнаружение всех областей с номерами
        /// </summary>
        /// <param name="countours"></param>
        /// <param name="hierarchy"></param>
        /// <param name="index"></param>
        /// <param name="gray"></param>
        /// <param name="canny"></param>
        /// <param name="licensePlateImageList"></param>
        /// <param name="filteredePlateImageList"></param>
        /// <param name="detectedPlateImageList"></param>
        /// <param name="licenses"></param>
        private void FindNum (VectorOfVectorOfPoint countours, int[,] hierarchy,
            int index, IInputArray gray, IInputArray canny,
            List<IInputOutputArray> licensePlateImageList,
            List<IInputOutputArray> filteredePlateImageList,
            List<RotatedRect> detectedPlateImageList,
            List<string>licenses)
        {
            for (; index >= 0; index = hierarchy[index, 0])
            {
                int numberOfChildre = GetNumberChildren(hierarchy, index);
                if (numberOfChildre == 0) continue;

                using (VectorOfPoint contour = countours[index])
                {
                    if (CvInvoke.ContourArea(contour) > 400)
                    {
                        if (numberOfChildre < 3) //проверка на 3 знака в регионе и опускание ниже
                        {
                            FindNum(countours, hierarchy, hierarchy[index, 2], gray, canny, licensePlateImageList,
                                filteredePlateImageList, detectedPlateImageList, licenses);

                            continue;
                        }

                        RotatedRect box = CvInvoke.MinAreaRect(contour);
                        if (box.Angle < -45.0)
                        {
                            float tmp = box.Size.Width;

                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;

                            box.Angle += 90.0f;
                        }
                        else if (box.Angle > 45.0)
                        {
                            float tmp = box.Size.Width;

                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;

                            box.Angle -= 90.0f;
                        }

                        double whRatio = (double)box.Size.Width / box.Size.Height;
                        if (!(3.0 < whRatio && whRatio < 10.0)) //Если значение не попало в условие, значит знака нет и запускает рекурсию
                        {
                            if (hierarchy[index, 2] > 0)
                            {
                                FindNum(countours, hierarchy, hierarchy[index, 2], gray, canny, licensePlateImageList,
                                    filteredePlateImageList, detectedPlateImageList, licenses);
                                continue;
                            }
                        }

                        using (UMat tmp1 = new UMat())
                        {
                            using (UMat tmp2 = new UMat())
                            {
                                PointF[] srcCorners = box.GetVertices();
                                PointF[] dstCorners = new PointF[]
                                {
                                    new PointF(0, box.Size.Height - 1),
                                    new PointF(0, 0),
                                    new PointF(box.Size.Width - 1, 0),
                                    new PointF(box.Size.Width - 1, box.Size.Height - 1)
                                };

                                using (Mat rot = CvInvoke.GetAffineTransform(srcCorners, dstCorners))
                                {
                                    CvInvoke.WarpAffine(gray, tmp1, rot, Size.Round(box.Size));
                                }

                                //Изменяем размер региона для точности определения через OCR
                                Size approxeSize = new Size(240, 180);
                                double scale = Math.Min(approxeSize.Width / box.Size.Width,
                                    approxeSize.Height / box.Size.Height);

                                Size newSize = new Size((int)Math.Round(box.Size.Width * scale),
                                    (int)Math.Round(box.Size.Height * scale));

                                CvInvoke.Resize(tmp1, tmp2, newSize, 0, 0, Inter.Cubic);

                                int edgePicelSize = 3;
                                Rectangle newRoi = new Rectangle(new Point(edgePicelSize, edgePicelSize),
                                    tmp2.Size - new Size(2 * edgePicelSize, 2 * edgePicelSize));

                                UMat plate = new UMat(tmp2, newRoi);
                                UMat filteredPlate = FilterPlate(plate);
                                StringBuilder stringBuilder = new StringBuilder();

                                using (UMat tmp = filteredPlate.Clone())
                                {
                                    OCR.SetImage(tmp);
                                    OCR.Recognize(); //Распознаем весь текст на переданом изображении
                                    stringBuilder.Append(OCR.GetUTF8Text()); //Передаем распознанный текст
                                    licenses.Add(stringBuilder.ToString());
                                    licensePlateImageList.Add(plate);
                                    filteredePlateImageList.Add(filteredPlate);
                                    detectedPlateImageList.Add(box);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Определение наличия номера в нужном регионе
        /// </summary>
        /// <param name="hierarchy"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        private int GetNumberChildren(int[,] hierarchy, int index)
        {
            try
            {
                index = hierarchy[index, 2];
                if (index < 0)
                {
                    return 0;
                }

                int count = 1;
                while (hierarchy[index, 0] > 0)
                {
                    count++;
                    index = hierarchy[index, 0];
                }

                return count;
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// Фильтрация изображения
        /// </summary>
        /// <param name="plate">Изображение</param>
        /// <returns>Изображение без шума</returns>
        private static UMat FilterPlate(UMat plate)
        {
            try
            {
                UMat result = new UMat();
                CvInvoke.Threshold(plate, result, 120, 255, ThresholdType.Trunc);
                Size plateSize = plate.Size;

                using (Mat plateMask = new Mat(plateSize.Height, plateSize.Width, DepthType.Cv8U, 1))
                {
                    using (Mat plateCanny = new Mat())
                    {
                        using (VectorOfVectorOfPoint counturs = new VectorOfVectorOfPoint())
                        {
                            plateMask.SetTo(new MCvScalar(255.0));
                            CvInvoke.Canny(plate, plateCanny, 100, 50);
                            CvInvoke.FindContours(plateCanny, counturs, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                            int count = counturs.Size;
                            for (int i = 0; i < count; i++)
                            {
                                using (VectorOfPoint contour = counturs[i])
                                {
                                    Rectangle rect = CvInvoke.BoundingRectangle(contour);

                                    if (rect.Height > (plateSize.Height >> 1))
                                    {
                                        rect.X -= 1;
                                        rect.Y -= 1;
                                        rect.Width += 2;
                                        rect.Height += 2;

                                        Rectangle roi = new Rectangle(Point.Empty, plate.Size);
                                        rect.Intersect(roi);
                                        CvInvoke.Rectangle(plateMask, rect, new MCvScalar(), -1);
                                    }
                                }
                            }
                            result.SetTo(new MCvScalar(), plateMask);
                        }
                    }
                }

                CvInvoke.Erode(result, result, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);
                CvInvoke.Dilate(result, result, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

                return result;
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Очистка ресурсов
        /// </summary>
        protected override void DisposeObject()
        {
            OCR.Dispose();
        }
    }
}
