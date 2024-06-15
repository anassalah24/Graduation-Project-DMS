#ifndef CONFIGSWIDGET_H
#define CONFIGSWIDGET_H

#include <QWidget>
#include "datahandler.h"
#include <QMessageBox>
#include "connectionwidget.h"
#include "systemcontrol.h"


namespace Ui {
class ConfigsWidget;
}

class ConfigsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConfigsWidget(DataHandler *dataHandler,ConnectionWidget *connectionWidget,SystemControl *systemControl, QWidget *parent = nullptr);
    ~ConfigsWidget();

private slots:
    void onCaptureSourceChanged(int index);
    void onApplyChangesClicked();

private:
    Ui::ConfigsWidget *ui;
    DataHandler *dataHandler;
    ConnectionWidget *connectionWidget;
    SystemControl *systemControl;
};

#endif // CONFIGSWIDGET_H
