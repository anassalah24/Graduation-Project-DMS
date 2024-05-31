#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "connectionwidget.h"
#include "datahandler.h"
#include "drivermonitoringwidget.h"
#include "systemcontrol.h"
#include "readingswidget.h"
#include "configswidget.h"
#include <opencv2/opencv.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    ConnectionWidget *connectionWidget;
    DataHandler *dataHandler;
    DriverMonitoringWidget *driverMonitoringWidget;
    SystemControl *systemcontrol;
    ReadingsWidget *readingsWidget;
    ConfigsWidget *configsWidget;


};

#endif // MAINWINDOW_H
