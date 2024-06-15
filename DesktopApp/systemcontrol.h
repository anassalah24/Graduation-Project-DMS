#ifndef SYSTEMCONTROL_H
#define SYSTEMCONTROL_H

#include <QWidget>
#include "datahandler.h"
#include <QMessageBox>
#include "connectionwidget.h"


namespace Ui {
class SystemControl;
}

class SystemControl : public QWidget
{
    Q_OBJECT

public:
    explicit SystemControl(DataHandler *dataHandler,ConnectionWidget *connectionWidget, QWidget *parent = nullptr);
    ~SystemControl();
    bool systemStatus=true;


private slots:
    void onSystemOnClicked();
    void onSystemOffClicked();
    void onSendButtonClicked();

private:
    Ui::SystemControl *ui;
    DataHandler *dataHandler;
    ConnectionWidget *connectionWidget;

};

#endif // SYSTEMCONTROL_H
