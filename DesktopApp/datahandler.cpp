#include "datahandler.h"
#include <QDataStream>
#include <QBuffer>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

DataHandler::DataHandler(QTcpSocket *socket1, QTcpSocket *socket2, QObject *parent) : QObject(parent), tcpSocket1(socket1), tcpSocket2(socket2) {
    connect(tcpSocket1, &QTcpSocket::readyRead, this, &DataHandler::onDataReady1);
    connect(tcpSocket2, &QTcpSocket::readyRead, this, &DataHandler::onDataReady2);
}

void DataHandler::onDataReady1() {
    QDataStream in(tcpSocket1);
    in.setVersion(QDataStream::Qt_5_15);

    quint32 imgSize;
    if (tcpSocket1->bytesAvailable() < (int)sizeof(quint32))
        return;
    in >> imgSize;

    QByteArray buffer;
    while (tcpSocket1->bytesAvailable() < imgSize)
        tcpSocket1->waitForReadyRead();
    buffer = tcpSocket1->read(imgSize);

    std::vector<uchar> data(buffer.begin(), buffer.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);

    if (!img.empty()) {
        QImage faceImage = matToQImage(img);
        emit faceReceived(faceImage);
    }
}

void DataHandler::onDataReady2() {
    // Read all available data from the second socket
    QByteArray data = tcpSocket2->readAll();

    // Print the received data to the console
    qDebug() << "Received data on socket 2:" << data;

    // Here, you can implement additional logic to handle the received data
}

void DataHandler::sendData(const QByteArray &data) {
    if (tcpSocket2->state() == QAbstractSocket::ConnectedState) {
        tcpSocket2->write(data);
        tcpSocket2->flush();
    } else {
        qDebug() << "Socket 2 is not connected!";
    }
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
