#include "drivermonitoringwidget.h"
#include "ui_drivermonitoringwidget.h"

DriverMonitoringWidget::DriverMonitoringWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::DriverMonitoringWidget)
{
    ui->setupUi(this);
}

DriverMonitoringWidget::~DriverMonitoringWidget()
{
    delete ui;
}

void DriverMonitoringWidget::updateFaceDisplay(QImage faceImage)
{
    ui->faceLabel->setPixmap(QPixmap::fromImage(faceImage));
}
