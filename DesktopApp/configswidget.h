#ifndef CONFIGSWIDGET_H
#define CONFIGSWIDGET_H

#include <QWidget>
#include "datahandler.h"

namespace Ui {
class ConfigsWidget;
}

class ConfigsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ConfigsWidget(DataHandler *dataHandler, QWidget *parent = nullptr);
    ~ConfigsWidget();

private slots:
    void onCaptureSourceChanged(int index);
    void onApplyChangesClicked();

private:
    Ui::ConfigsWidget *ui;
    DataHandler *dataHandler;
};

#endif // CONFIGSWIDGET_H
