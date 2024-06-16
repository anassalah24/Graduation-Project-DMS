#ifndef DRIVERMONITORINGWIDGET_H
#define DRIVERMONITORINGWIDGET_H

#include <QImage>
#include <QWidget>
#include "datahandler.h"
#include "connectionwidget.h"
#include "systemcontrol.h"
#include <QPainter>


namespace Ui {
class DriverMonitoringWidget;
}

class DriverMonitoringWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DriverMonitoringWidget(DataHandler *dataHandler, ConnectionWidget *connectionWidget,SystemControl *systemControl,QWidget *parent = nullptr);
    ~DriverMonitoringWidget();

public slots:
    void updateFaceDisplay(QImage faceImage);

private:
    Ui::DriverMonitoringWidget *ui;
    DataHandler *dataHandler;
    ConnectionWidget *connectionWidget;
    SystemControl *systemControl;
};

#endif // DRIVERMONITORINGWIDGET_H
