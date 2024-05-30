#ifndef DRIVERMONITORINGWIDGET_H
#define DRIVERMONITORINGWIDGET_H

#include <QImage>
#include <QWidget>

namespace Ui {
class DriverMonitoringWidget;
}

class DriverMonitoringWidget : public QWidget
{
    Q_OBJECT

public:
    explicit DriverMonitoringWidget(QWidget *parent = nullptr);
    ~DriverMonitoringWidget();

public slots:
    void updateFaceDisplay(QImage faceImage);

private:
    Ui::DriverMonitoringWidget *ui;
};

#endif // DRIVERMONITORINGWIDGET_H
