#include "datahandler.h"
#include <QBuffer>

DataHandler::DataHandler(QTcpSocket *socket, QObject *parent) : QObject(parent), tcpSocket(socket) {
    connect(tcpSocket, &QTcpSocket::readyRead, this, &DataHandler::onDataReady);
}

void DataHandler::onDataReady() {
    // QByteArray data = tcpSocket->readAll();
    // QBuffer buffer(&data);
    // buffer.open(QIODevice::ReadOnly);
    // QImage faceImage;
    // faceImage.load(&buffer, "JPEG");

    // if (!faceImage.isNull()) {
    //     emit faceReceived(faceImage);
    // }
    QDataStream in(tcpSocket);
    in.setVersion(QDataStream::Qt_5_15);

    quint32 imgSize;
    if (tcpSocket->bytesAvailable() < (int)sizeof(quint32))
        return;
    in >> imgSize;

    QByteArray buffer;
    while (tcpSocket->bytesAvailable() < imgSize)
        tcpSocket->waitForReadyRead();
    buffer = tcpSocket->read(imgSize);

    std::vector<uchar> data(buffer.begin(), buffer.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
    QImage faceImage = matToQImage(img);
    emit faceReceived(faceImage);
}

QImage DataHandler::matToQImage(const cv::Mat &mat) {
    switch (mat.type()) {
    case CV_8UC1:
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
    case CV_8UC3:
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped();
    case CV_8UC4:
        return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
    default:
        return QImage();
    }
}


