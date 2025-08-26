# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'scanSpaceImageProcessor_settings.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QCheckBox, QComboBox,
    QDialog, QDialogButtonBox, QFrame, QGridLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpacerItem, QSpinBox, QTabWidget,
    QWidget)

class Ui_ImageProcessorSettings(object):
    def setupUi(self, ImageProcessorSettings):
        if not ImageProcessorSettings.objectName():
            ImageProcessorSettings.setObjectName(u"ImageProcessorSettings")
        ImageProcessorSettings.resize(595, 482)
        self.gridLayout_2 = QGridLayout(ImageProcessorSettings)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.tabWidget = QTabWidget(ImageProcessorSettings)
        self.tabWidget.setObjectName(u"tabWidget")
        self.generalSettingsTab = QWidget()
        self.generalSettingsTab.setObjectName(u"generalSettingsTab")
        self.gridLayout_5 = QGridLayout(self.generalSettingsTab)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.applicationGeneralGridLayout = QGridLayout()
        self.applicationGeneralGridLayout.setObjectName(u"applicationGeneralGridLayout")
        self.label_12 = QLabel(self.generalSettingsTab)
        self.label_12.setObjectName(u"label_12")

        self.applicationGeneralGridLayout.addWidget(self.label_12, 18, 0, 1, 1)

        self.chartFolderPathLineEdit = QLineEdit(self.generalSettingsTab)
        self.chartFolderPathLineEdit.setObjectName(u"chartFolderPathLineEdit")

        self.applicationGeneralGridLayout.addWidget(self.chartFolderPathLineEdit, 19, 1, 1, 1)

        self.line_2 = QFrame(self.generalSettingsTab)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        self.applicationGeneralGridLayout.addWidget(self.line_2, 13, 0, 1, 3)

        self.label_3 = QLabel(self.generalSettingsTab)
        self.label_3.setObjectName(u"label_3")

        self.applicationGeneralGridLayout.addWidget(self.label_3, 17, 0, 1, 1)

        self.logDebugDepthCombobox = QComboBox(self.generalSettingsTab)
        self.logDebugDepthCombobox.setObjectName(u"logDebugDepthCombobox")
        self.logDebugDepthCombobox.setEnabled(True)
        self.logDebugDepthCombobox.setMinimumSize(QSize(100, 0))

        self.applicationGeneralGridLayout.addWidget(self.logDebugDepthCombobox, 2, 1, 1, 1, Qt.AlignLeft)

        self.label_11 = QLabel(self.generalSettingsTab)
        self.label_11.setObjectName(u"label_11")
        font = QFont()
        font.setBold(True)
        self.label_11.setFont(font)
        self.label_11.setAlignment(Qt.AlignCenter)

        self.applicationGeneralGridLayout.addWidget(self.label_11, 14, 0, 1, 3)

        self.exifCopywriteLineEdit = QLineEdit(self.generalSettingsTab)
        self.exifCopywriteLineEdit.setObjectName(u"exifCopywriteLineEdit")
        self.exifCopywriteLineEdit.setEnabled(False)

        self.applicationGeneralGridLayout.addWidget(self.exifCopywriteLineEdit, 20, 2, 1, 1)

        self.label = QLabel(self.generalSettingsTab)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 20))
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        self.applicationGeneralGridLayout.addWidget(self.label, 0, 0, 1, 3)

        self.defaultColorspaceComboBox = QComboBox(self.generalSettingsTab)
        self.defaultColorspaceComboBox.setObjectName(u"defaultColorspaceComboBox")

        self.applicationGeneralGridLayout.addWidget(self.defaultColorspaceComboBox, 18, 1, 1, 1)

        self.defaultThreadCountSpinbox = QSpinBox(self.generalSettingsTab)
        self.defaultThreadCountSpinbox.setObjectName(u"defaultThreadCountSpinbox")
        self.defaultThreadCountSpinbox.setMaximumSize(QSize(50, 16777215))
        self.defaultThreadCountSpinbox.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.defaultThreadCountSpinbox.setMinimum(1)
        self.defaultThreadCountSpinbox.setMaximum(100)
        self.defaultThreadCountSpinbox.setValue(4)

        self.applicationGeneralGridLayout.addWidget(self.defaultThreadCountSpinbox, 3, 1, 1, 1, Qt.AlignLeft)

        self.label_4 = QLabel(self.generalSettingsTab)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setEnabled(False)

        self.applicationGeneralGridLayout.addWidget(self.label_4, 2, 0, 1, 1)

        self.defaultExportFormatComboBox = QComboBox(self.generalSettingsTab)
        self.defaultExportFormatComboBox.setObjectName(u"defaultExportFormatComboBox")
        self.defaultExportFormatComboBox.setMaximumSize(QSize(60, 60))
        self.defaultExportFormatComboBox.setLayoutDirection(Qt.RightToLeft)

        self.applicationGeneralGridLayout.addWidget(self.defaultExportFormatComboBox, 17, 1, 1, 1, Qt.AlignRight)

        self.allowChartlessProcessingCheckbox = QCheckBox(self.generalSettingsTab)
        self.allowChartlessProcessingCheckbox.setObjectName(u"allowChartlessProcessingCheckbox")

        self.applicationGeneralGridLayout.addWidget(self.allowChartlessProcessingCheckbox, 16, 1, 1, 2)

        self.browseForDefaultColorChartPushButton = QPushButton(self.generalSettingsTab)
        self.browseForDefaultColorChartPushButton.setObjectName(u"browseForDefaultColorChartPushButton")

        self.applicationGeneralGridLayout.addWidget(self.browseForDefaultColorChartPushButton, 19, 2, 1, 1)

        self.exifTagsLineEdit = QLineEdit(self.generalSettingsTab)
        self.exifTagsLineEdit.setObjectName(u"exifTagsLineEdit")
        self.exifTagsLineEdit.setEnabled(False)

        self.applicationGeneralGridLayout.addWidget(self.exifTagsLineEdit, 22, 2, 1, 1)

        self.label_8 = QLabel(self.generalSettingsTab)
        self.label_8.setObjectName(u"label_8")

        self.applicationGeneralGridLayout.addWidget(self.label_8, 3, 0, 1, 1)

        self.embedCustomEXIFCheckbox = QCheckBox(self.generalSettingsTab)
        self.embedCustomEXIFCheckbox.setObjectName(u"embedCustomEXIFCheckbox")
        self.embedCustomEXIFCheckbox.setEnabled(False)
        self.embedCustomEXIFCheckbox.setTristate(False)

        self.applicationGeneralGridLayout.addWidget(self.embedCustomEXIFCheckbox, 20, 0, 1, 2)

        self.bitDepth16EnableCheckbox = QCheckBox(self.generalSettingsTab)
        self.bitDepth16EnableCheckbox.setObjectName(u"bitDepth16EnableCheckbox")

        self.applicationGeneralGridLayout.addWidget(self.bitDepth16EnableCheckbox, 17, 2, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.applicationGeneralGridLayout.addItem(self.verticalSpacer_2, 24, 1, 1, 1)

        self.enableDarkThemeCheckbox = QCheckBox(self.generalSettingsTab)
        self.enableDarkThemeCheckbox.setObjectName(u"enableDarkThemeCheckbox")
        self.enableDarkThemeCheckbox.setEnabled(True)

        self.applicationGeneralGridLayout.addWidget(self.enableDarkThemeCheckbox, 5, 2, 1, 1)

        self.usePrecalculatedChartsCheckBox = QCheckBox(self.generalSettingsTab)
        self.usePrecalculatedChartsCheckBox.setObjectName(u"usePrecalculatedChartsCheckBox")

        self.applicationGeneralGridLayout.addWidget(self.usePrecalculatedChartsCheckBox, 19, 0, 1, 1)

        self.displayLogCheckBox = QCheckBox(self.generalSettingsTab)
        self.displayLogCheckBox.setObjectName(u"displayLogCheckBox")
        self.displayLogCheckBox.setChecked(True)

        self.applicationGeneralGridLayout.addWidget(self.displayLogCheckBox, 2, 2, 1, 1)

        self.exifAuthorsLineEdit = QLineEdit(self.generalSettingsTab)
        self.exifAuthorsLineEdit.setObjectName(u"exifAuthorsLineEdit")
        self.exifAuthorsLineEdit.setEnabled(False)

        self.applicationGeneralGridLayout.addWidget(self.exifAuthorsLineEdit, 21, 2, 1, 1)

        self.colorCorrectThumbnailsCheckbox = QCheckBox(self.generalSettingsTab)
        self.colorCorrectThumbnailsCheckbox.setObjectName(u"colorCorrectThumbnailsCheckbox")

        self.applicationGeneralGridLayout.addWidget(self.colorCorrectThumbnailsCheckbox, 5, 1, 1, 1)


        self.gridLayout_5.addLayout(self.applicationGeneralGridLayout, 0, 0, 1, 1)

        self.tabWidget.addTab(self.generalSettingsTab, "")
        self.importExportRules = QWidget()
        self.importExportRules.setObjectName(u"importExportRules")
        self.gridLayout_6 = QGridLayout(self.importExportRules)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.importExportSettingsGridLayout = QGridLayout()
        self.importExportSettingsGridLayout.setObjectName(u"importExportSettingsGridLayout")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.groupImagesByPrefixCheckbox = QCheckBox(self.importExportRules)
        self.groupImagesByPrefixCheckbox.setObjectName(u"groupImagesByPrefixCheckbox")

        self.gridLayout.addWidget(self.groupImagesByPrefixCheckbox, 2, 0, 1, 1)

        self.prefixGroupingLineEdit = QLineEdit(self.importExportRules)
        self.prefixGroupingLineEdit.setObjectName(u"prefixGroupingLineEdit")

        self.gridLayout.addWidget(self.prefixGroupingLineEdit, 2, 1, 1, 1)

        self.groupImagesBySubfolderCheckbox = QCheckBox(self.importExportRules)
        self.groupImagesBySubfolderCheckbox.setObjectName(u"groupImagesBySubfolderCheckbox")

        self.gridLayout.addWidget(self.groupImagesBySubfolderCheckbox, 1, 0, 1, 1)

        self.lookForImagesInSubfolderCheckbox = QCheckBox(self.importExportRules)
        self.lookForImagesInSubfolderCheckbox.setObjectName(u"lookForImagesInSubfolderCheckbox")

        self.gridLayout.addWidget(self.lookForImagesInSubfolderCheckbox, 0, 0, 1, 1)

        self.ignoreFormatsCheckbox = QCheckBox(self.importExportRules)
        self.ignoreFormatsCheckbox.setObjectName(u"ignoreFormatsCheckbox")

        self.gridLayout.addWidget(self.ignoreFormatsCheckbox, 3, 0, 1, 1)

        self.ignoreStringLineEdit = QLineEdit(self.importExportRules)
        self.ignoreStringLineEdit.setObjectName(u"ignoreStringLineEdit")

        self.gridLayout.addWidget(self.ignoreStringLineEdit, 3, 1, 1, 1)


        self.importExportSettingsGridLayout.addLayout(self.gridLayout, 1, 2, 1, 1)

        self.exportRulesTextBox = QPlainTextEdit(self.importExportRules)
        self.exportRulesTextBox.setObjectName(u"exportRulesTextBox")
        self.exportRulesTextBox.setMinimumSize(QSize(0, 140))
        self.exportRulesTextBox.setMaximumSize(QSize(16777215, 140))
        self.exportRulesTextBox.setFrameShape(QFrame.StyledPanel)
        self.exportRulesTextBox.setUndoRedoEnabled(False)
        self.exportRulesTextBox.setReadOnly(True)
        self.exportRulesTextBox.setTextInteractionFlags(Qt.NoTextInteraction)
        self.exportRulesTextBox.setBackgroundVisible(False)

        self.importExportSettingsGridLayout.addWidget(self.exportRulesTextBox, 6, 2, 1, 1)

        self.useExportRulesCheckbox = QCheckBox(self.importExportRules)
        self.useExportRulesCheckbox.setObjectName(u"useExportRulesCheckbox")

        self.importExportSettingsGridLayout.addWidget(self.useExportRulesCheckbox, 4, 1, 1, 1)

        self.exportSettingsLineEdit = QLineEdit(self.importExportRules)
        self.exportSettingsLineEdit.setObjectName(u"exportSettingsLineEdit")

        self.importExportSettingsGridLayout.addWidget(self.exportSettingsLineEdit, 4, 2, 1, 1)

        self.useImportRulesCheckbox = QCheckBox(self.importExportRules)
        self.useImportRulesCheckbox.setObjectName(u"useImportRulesCheckbox")

        self.importExportSettingsGridLayout.addWidget(self.useImportRulesCheckbox, 1, 1, 1, 1)

        self.label_6 = QLabel(self.importExportRules)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setFont(font)
        self.label_6.setAlignment(Qt.AlignCenter)

        self.importExportSettingsGridLayout.addWidget(self.label_6, 2, 1, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.importExportSettingsGridLayout.addItem(self.verticalSpacer, 7, 1, 1, 1)

        self.label_5 = QLabel(self.importExportRules)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.importExportSettingsGridLayout.addWidget(self.label_5, 0, 1, 1, 2)

        self.schema_preview_label = QLabel(self.importExportRules)
        self.schema_preview_label.setObjectName(u"schema_preview_label")

        self.importExportSettingsGridLayout.addWidget(self.schema_preview_label, 5, 2, 1, 1)


        self.gridLayout_6.addLayout(self.importExportSettingsGridLayout, 0, 0, 1, 1)

        self.tabWidget.addTab(self.importExportRules, "")
        self.serverSettingsTab = QWidget()
        self.serverSettingsTab.setObjectName(u"serverSettingsTab")
        self.gridLayout_8 = QGridLayout(self.serverSettingsTab)
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.serverSettingsGridLayout = QGridLayout()
        self.serverSettingsGridLayout.setObjectName(u"serverSettingsGridLayout")
        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.serverSettingsGridLayout.addItem(self.verticalSpacer_4, 5, 1, 1, 1)

        self.hostServerAddressLineEdit = QLineEdit(self.serverSettingsTab)
        self.hostServerAddressLineEdit.setObjectName(u"hostServerAddressLineEdit")

        self.serverSettingsGridLayout.addWidget(self.hostServerAddressLineEdit, 0, 1, 1, 2)

        self.openServerHelpPushButton = QPushButton(self.serverSettingsTab)
        self.openServerHelpPushButton.setObjectName(u"openServerHelpPushButton")

        self.serverSettingsGridLayout.addWidget(self.openServerHelpPushButton, 1, 2, 1, 1)

        self.enableServerCheckbox = QCheckBox(self.serverSettingsTab)
        self.enableServerCheckbox.setObjectName(u"enableServerCheckbox")
        font1 = QFont()
        font1.setPointSize(10)
        font1.setBold(True)
        self.enableServerCheckbox.setFont(font1)
        self.enableServerCheckbox.setIconSize(QSize(20, 20))

        self.serverSettingsGridLayout.addWidget(self.enableServerCheckbox, 0, 0, 1, 1)


        self.gridLayout_8.addLayout(self.serverSettingsGridLayout, 1, 0, 1, 1)

        self.tabWidget.addTab(self.serverSettingsTab, "")

        self.gridLayout_2.addWidget(self.tabWidget, 2, 0, 1, 1)

        self.finalizeSettingsButtonBox = QDialogButtonBox(ImageProcessorSettings)
        self.finalizeSettingsButtonBox.setObjectName(u"finalizeSettingsButtonBox")
        self.finalizeSettingsButtonBox.setOrientation(Qt.Horizontal)
        self.finalizeSettingsButtonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)

        self.gridLayout_2.addWidget(self.finalizeSettingsButtonBox, 3, 0, 1, 1)


        self.retranslateUi(ImageProcessorSettings)
        self.finalizeSettingsButtonBox.accepted.connect(ImageProcessorSettings.accept)
        self.finalizeSettingsButtonBox.rejected.connect(ImageProcessorSettings.reject)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(ImageProcessorSettings)
    # setupUi

    def retranslateUi(self, ImageProcessorSettings):
        ImageProcessorSettings.setWindowTitle(QCoreApplication.translate("ImageProcessorSettings", u"Image Processor Settings", None))
        self.label_12.setText(QCoreApplication.translate("ImageProcessorSettings", u"Default Colorspace", None))
        self.chartFolderPathLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Path", None))
        self.label_3.setText(QCoreApplication.translate("ImageProcessorSettings", u"Default Export Format", None))
        self.label_11.setText(QCoreApplication.translate("ImageProcessorSettings", u"Image Processing Settings", None))
        self.exifCopywriteLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Copywrite", None))
        self.label.setText(QCoreApplication.translate("ImageProcessorSettings", u"Application General Settings", None))
#if QT_CONFIG(tooltip)
        self.defaultColorspaceComboBox.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Sets the default colourspace where it applies", None))
#endif // QT_CONFIG(tooltip)
        self.defaultThreadCountSpinbox.setSuffix("")
        self.label_4.setText(QCoreApplication.translate("ImageProcessorSettings", u"Log Debug Depth", None))
        self.defaultExportFormatComboBox.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"jpg", None))
#if QT_CONFIG(tooltip)
        self.allowChartlessProcessingCheckbox.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"If no chart found, processes and exports images as is", None))
#endif // QT_CONFIG(tooltip)
        self.allowChartlessProcessingCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Allow Processing of Images without Chart", None))
        self.browseForDefaultColorChartPushButton.setText(QCoreApplication.translate("ImageProcessorSettings", u"Browse", None))
        self.exifTagsLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Tags", None))
        self.label_8.setText(QCoreApplication.translate("ImageProcessorSettings", u"Default thread count", None))
#if QT_CONFIG(tooltip)
        self.embedCustomEXIFCheckbox.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Embeds EXIF data into the exported images. uses image defaults if nothing entered", None))
#endif // QT_CONFIG(tooltip)
        self.embedCustomEXIFCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Embed Custom EXIF", None))
#if QT_CONFIG(tooltip)
        self.bitDepth16EnableCheckbox.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Set the default processing to 16 bit if the format allows", None))
#endif // QT_CONFIG(tooltip)
        self.bitDepth16EnableCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"16 Bit Default", None))
#if QT_CONFIG(tooltip)
        self.enableDarkThemeCheckbox.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.enableDarkThemeCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Dark Theme", None))
        self.usePrecalculatedChartsCheckBox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Use Precalculated Charts", None))
        self.displayLogCheckBox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Display Log", None))
        self.exifAuthorsLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Authors", None))
        self.colorCorrectThumbnailsCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Display thumbnails with colour correction", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.generalSettingsTab), QCoreApplication.translate("ImageProcessorSettings", u"General Settings", None))
#if QT_CONFIG(tooltip)
        self.groupImagesByPrefixCheckbox.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Groups images by filename prefix, for instance \"CAM01_0001\" will be grouped under \"CAM01\"", None))
#endif // QT_CONFIG(tooltip)
        self.groupImagesByPrefixCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Group images by Filename Prefix", None))
#if QT_CONFIG(tooltip)
        self.prefixGroupingLineEdit.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Groups images ", None))
#endif // QT_CONFIG(tooltip)
        self.prefixGroupingLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Seperator", None))
        self.groupImagesBySubfolderCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Group Images by Subfolders", None))
        self.lookForImagesInSubfolderCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Look for images in Subfolders", None))
        self.ignoreFormatsCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Ignore Filter", None))
#if QT_CONFIG(tooltip)
        self.ignoreStringLineEdit.setToolTip(QCoreApplication.translate("ImageProcessorSettings", u"Ignore Images with matching text. Use commas to ignore multiple types. Usage: \"REFERENCE\" will ignore all images with REFERENCE in the filename.", None))
#endif // QT_CONFIG(tooltip)
        self.ignoreStringLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Ignore String, Use commas for Multiple", None))
        self.exportRulesTextBox.setPlainText(QCoreApplication.translate("ImageProcessorSettings", u"[r]: Root folder name\n"
"[s]: Sub folder name (if exists)\n"
"[e]: File name extension\n"
"[oc]: Original file name without numbers\n"
"[c]: Custom name from custom input\n"
"[o]: Original file name\n"
"[n]: Image Number\n"
"[/]: new folder layer", None))
        self.useExportRulesCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Use Export Rules", None))
        self.exportSettingsLineEdit.setText(QCoreApplication.translate("ImageProcessorSettings", u"[r]/[o][n4][e]", None))
        self.useImportRulesCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Use Import Rules", None))
        self.label_6.setText(QCoreApplication.translate("ImageProcessorSettings", u"Export Settings", None))
        self.label_5.setText(QCoreApplication.translate("ImageProcessorSettings", u"Import Settings", None))
        self.schema_preview_label.setText(QCoreApplication.translate("ImageProcessorSettings", u"Schema Preview", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.importExportRules), QCoreApplication.translate("ImageProcessorSettings", u"Import and Export", None))
        self.hostServerAddressLineEdit.setPlaceholderText(QCoreApplication.translate("ImageProcessorSettings", u"Server address", None))
        self.openServerHelpPushButton.setText(QCoreApplication.translate("ImageProcessorSettings", u"Help", None))
        self.enableServerCheckbox.setText(QCoreApplication.translate("ImageProcessorSettings", u"Enable Server Connections", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.serverSettingsTab), QCoreApplication.translate("ImageProcessorSettings", u"Server Settings", None))
    # retranslateUi

