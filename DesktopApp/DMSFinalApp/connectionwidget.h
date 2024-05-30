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
    QTcpSocket *tcpSocket;
    ~ConnectionWidget();

signals:
    void connected();
    void disconnected();

private slots:
    void onConnectButtonClicked();
    void onDisconnectButtonClicked();
    void onConnected();
    void onDisconnected();

private:
    Ui::ConnectionWidget *ui;

};

#endif // CONNECTIONWIDGET_H
