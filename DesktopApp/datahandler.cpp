#include "datahandler.h"
#include <QDataStream>
#include <QBuffer>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <QDebug>

DataHandler::DataHandler(QTcpSocket *socket1, QTcpSocket *socket2, QObject *parent) : QObject(parent), tcpSocket1(socket1), tcpSocket2(socket2) {
    connect(tcpSocket1, &QTcpSocket::readyRead, this, &DataHandler::onDataReady1);
    connect(tcpSocket2, &QTcpSocket::readyRead, this, &DataHandler::onDataReady2);

    frameCheckTimer = new QTimer(this);
    connect(frameCheckTimer, &QTimer::timeout, this, &DataHandler::checkFrameReception);
    frameCheckTimer->start(1000); // Check every second

    readingsCheckTimer = new QTimer(this);
    connect(readingsCheckTimer, &QTimer::timeout, this, &DataHandler::checkReadingsReception);
    readingsCheckTimer->start(1000); // Check every second
}

DataHandler::~DataHandler() {
    frameCheckTimer->stop();
}


void DataHandler::onDataReady1() {
    frameReceivedSinceLastCheck = true;
    QDataStream in(tcpSocket1);
    in.setVersion(QDataStream::Qt_5_15);

    quint32 imgSize;
    if (tcpSocket1->bytesAvailable() < (int)sizeof(quint32))
        return;
    in >> imgSize;

    if (imgSize == 0) {
        // If no image size, emit empty QImage
        emit faceReceived(QImage());
        return;
    }

    QByteArray buffer;
    while (tcpSocket1->bytesAvailable() < imgSize)
        tcpSocket1->waitForReadyRead();
    buffer = tcpSocket1->read(imgSize);

    std::vector<uchar> data(buffer.begin(), buffer.end());
    cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);

    if (img.empty()) {
        // If decoding fails or image is empty, also emit empty QImage
        emit faceReceived(QImage());
    } else {
        QImage faceImage = matToQImage(img);
        emit faceReceived(faceImage);
    }
}


void DataHandler::onDataReady2() {
    readingsReceivedSinceLastCheck = true;
    // Read all available data from the second socket
    QByteArray data = tcpSocket2->readAll();

    // Convert QByteArray to std::vector<uint8_t>
    std::vector<uint8_t> buffer(data.begin(), data.end());

    // Deserialize the data
    std::vector<std::vector<float>> readings = deserialize(buffer);

    // Print the received data to the console
    qDebug() << "Received data on socket 2:" << data;
    qDebug() << "Deserialized readings:";

    for (const auto& row : readings) {
        QString rowStr;
        for (const auto& value : row) {
            rowStr += QString::number(value) + " ";
        }
        qDebug() << rowStr;
    }
    // Emit the readingsReceived signal with the deserialized data
    emit readingsReceived(readings);
    // Here, you can implement additional logic to handle the deserialized data
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


// Function to deserialize a byte array to a 2D vector of floats
std::vector<std::vector<float>> DataHandler::deserialize(const std::vector<uint8_t>& buffer) {
    size_t rows, cols;
    const uint8_t* ptr = buffer.data();

    // Copy rows and cols from the buffer
    std::memcpy(&rows, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    std::memcpy(&cols, ptr, sizeof(size_t));
    ptr += sizeof(size_t);

    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));

    // Copy each float value from the buffer
    for (auto& row : data) {
        std::memcpy(row.data(), ptr, row.size() * sizeof(float));
        ptr += row.size() * sizeof(float);
    }

    return data;
}

void DataHandler::checkFrameReception() {
    if (!frameReceivedSinceLastCheck) {
        // No frames received since last check
        emit faceReceived(QImage());
    }
    frameReceivedSinceLastCheck = false; // Reset the flag for the next interval
}

void DataHandler::checkReadingsReception() {
    if (!readingsReceivedSinceLastCheck) {
        // Emit a signal with default invalid data
        std::vector<std::vector<float>> invalidData(2, std::vector<float>(9, -100));
        emit readingsReceived(invalidData);
    }
    readingsReceivedSinceLastCheck = false; // Reset the flag
}

