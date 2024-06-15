#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include <QObject>
#include <QTcpSocket>
#include <QImage>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdint>  // For uint8_t
#include <QTimer>


class DataHandler : public QObject {
    Q_OBJECT

public:
    explicit DataHandler(QTcpSocket *socket1, QTcpSocket *socket2, QObject *parent = nullptr);
    ~DataHandler();  // Destructor declaration

    void sendData(const QByteArray &data);  // Method to send data

signals:
    void faceReceived(QImage faceImage);
    void readingsReceived(const std::vector<std::vector<float>>& readings);  // New signal for deserialized readings

private slots:
    void onDataReady1();
    void onDataReady2();
    void checkFrameReception();


private:
    QTcpSocket *tcpSocket1;
    QTcpSocket *tcpSocket2;
    QImage matToQImage(const cv::Mat &mat);
    std::vector<std::vector<float>> deserialize(const std::vector<uint8_t>& buffer);
    QTimer *frameCheckTimer;
    bool frameReceivedSinceLastCheck = false;
};

#endif // DATAHANDLER_H
