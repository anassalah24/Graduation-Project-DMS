#ifndef CONNECTIONWIDGET_H
#define CONNECTIONWIDGET_H

#include <QWidget>
#include <QTcpSocket>

namespace Ui {
class ConnectionWidget;
}

class ConnectionWidget : public QWidget {
    Q_OBJECT

public:
    explicit ConnectionWidget(QWidget *parent = nullptr);
    ~ConnectionWidget();
    QTcpSocket* getTcpSocket1() const;
    QTcpSocket* getTcpSocket2() const;
    bool isSystemConnected() const;

signals:
    void connected();
    void disconnected();

private slots:
    void onConnectButtonClicked();
    void onDisconnectButtonClicked();
    void onSocket1Connected();
    void onSocket2Connected();
    void onSocket1Disconnected();
    void onSocket2Disconnected();
    void onConnected();         // Add this line
    void onDisconnected();      // Add this line
    void onConnectionError(QAbstractSocket::SocketError socketError);

private:
    Ui::ConnectionWidget *ui;
    QTcpSocket *tcpSocket1;
    QTcpSocket *tcpSocket2;
    bool socket1Connected;
    bool socket2Connected;
};

#endif // CONNECTIONWIDGET_H
