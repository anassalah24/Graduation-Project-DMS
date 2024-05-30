#ifndef DATAHANDLER_H
#define DATAHANDLER_H

#include <QObject>
#include <QTcpSocket>
#include <QImage>
#include <opencv2/opencv.hpp>


class DataHandler : public QObject {
    Q_OBJECT

public:
    explicit DataHandler(QTcpSocket *socket, QObject *parent = nullptr);

signals:
    void faceReceived(QImage faceImage);

private slots:
    void onDataReady();

private:
    QTcpSocket *tcpSocket;
    QImage matToQImage(const cv::Mat &mat);

};

#endif // DATAHANDLER_H
