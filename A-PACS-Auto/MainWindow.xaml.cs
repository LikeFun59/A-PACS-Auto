using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using A_PACS_Auto.Classes;
using System.Drawing;

namespace A_PACS_Auto
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private ScanNum numRec;
        private Mat inputImage;
        public MainWindow()
        {
            InitializeComponent();
        }

        private void ProcessImage(IInputOutputArray image)
        {
            List<IInputOutputArray> licensesPlateImage = new List<IInputOutputArray>();
            List<IInputOutputArray> filteredPlateImage = new List<IInputOutputArray>();
            List<RotatedRect> rotations = new List<RotatedRect>();

            List<string> plate = numRec.DetectNum(image, licensesPlateImage, filteredPlateImage, rotations);

            leftStack.Children.Clear();
            for (int i = 0; i < plate.Count; i++)
            {
                Mat dest = new Mat();
                CvInvoke.VConcat(licensesPlateImage[i], filteredPlateImage[i], dest);
                InsertTextAndPic($"Номер: {plate[i]}", dest);
            }

            Image<Bgr, byte> outputImage = inputImage.ToImage<Bgr, byte>();

            foreach(RotatedRect rect in rotations)
            {
                PointF[] v = rect.GetVertices();

                PointF prevPoint = v[0];
                PointF firstPoint = prevPoint;
                PointF nextPoint = prevPoint;
                PointF lastPoint = nextPoint;

                for (int j = 1; j < v.Length; j++)
                {
                    nextPoint = v[j];
                    CvInvoke.Line(outputImage, System.Drawing.Point.Round(prevPoint), System.Drawing.Point.Round(nextPoint), new MCvScalar(0, 0, 255), 5, LineType.EightConnected, 0);
                    prevPoint = nextPoint;
                    lastPoint = prevPoint;
                }
            }

        }

        private void InsertTextAndPic (string text, IInputArray pic)
        {
            Label label = new Label();
            label.Content = text;
            label.Width = 100;
            label.Height = 30;
            leftStack.Children.Add(label);

            Image img = new Image();
            img = (Image)pic;
            leftStack.Children.Add(img);
        }

        private void btnUpload_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                OpenFileDialog open = new OpenFileDialog();
                open.Filter = "Image Files(*.jpg; *.jpeg; *.gif; *.bmp; *.png)|*.jpg; *.jpeg; *.gif; *.bmp; *.png";
                if (open.ShowDialog() == true)
                {
                    Uri uri = new Uri(open.FileName);
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.UriSource = uri;
                    imgViewer.Source = bitmap;
                    inputImage = new Mat(open.FileName);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Ошибка", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            numRec = new ScanNum("Resources", "rus");
        }

        private void btnScans_Click(object sender, RoutedEventArgs e)
        {
            UMat um = inputImage.GetUMat(AccessType.ReadWrite);
            ProcessImage(um);
        }
    }
}
