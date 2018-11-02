#include "BrainModeling.h"
#include <QFileDialog>
#include <time.h>
#include "ParallelModelingConsole.h"
#include "ShiPreprocessImage.h"


BrainModeling::BrainModeling(QWidget *parent, Qt::WindowFlags flags)
	: QWidget(parent, flags)
{
	ui.setupUi(this);
	pDynamicPETData=NULL;
	pMask=NULL;
	pAIF=NULL;
	pResultK1=NULL;
	pResultK2=NULL;
	pResultK3=NULL;
	pResultK4=NULL;
	pResultVB=NULL;

	pLabelMap=NULL;
	pParaList=NULL;
	pChiSquare=NULL;

	pointerData = NULL;
	imageMat = NULL;
	
	ui.radioButton_X->setChecked(false);
	ui.radioButton_Y->setChecked(false);
	ui.radioButton_Z->setChecked(true);
	ui.Button_Display->setCheckable(true);

	ui.ScrollBar_Slice->setUpdatesEnabled(true);
	ui.ScrollBar_Time->setUpdatesEnabled(true);
	ui.ScrollBar_Slice->setRange(1,20);
	ui.ScrollBar_Time->setRange(1,20);
	ui.ScrollBar_Slice->setValue(1);
	ui.ScrollBar_Time->setValue(1);	


	DPETFileName=new char[256];

	bSaveOTS=false;

	time(&CalculationEndTime);

	bExcuted=false;

	connect(ui.Button_LoadInput, SIGNAL(clicked()), this, SLOT(getInputPETDataPath()));
	connect(ui.Button_LoadIAIF, SIGNAL(clicked()), this, SLOT(getAIFPath()));
	connect(ui.Button_Computing, SIGNAL(clicked()), this, SLOT(onComputing()));
	connect(ui.CheckBox_IrreversibleFit, SIGNAL(clicked()),this, SLOT(updateIrreversibleFit()));

	connect(ui.ScrollBar_Slice, SIGNAL(valueChanged(int)), this, SLOT(z_axis(int)));
	connect(ui.ScrollBar_Time, SIGNAL(valueChanged(int)), this, SLOT(t_axis(int)));
	
	int NMaxCore=omp_get_max_threads();

	ui.ProgressBar_Computing->setValue(0);
	ui.CheckBox_WeightedFit->setChecked(true);
	ui.CheckBox_SmoothPET->setChecked(false);
	ui.CheckBox_Segmented->setChecked(false);
	ui.SpinBox_NumberCPUCore->setMaximum(NMaxCore);
	ui.SpinBox_NumberSupersample->setMaximum(64);
	ui.SpinBox_NumberCluster->setValue(8);
	ui.SpinBox_NumberCluster->setMaximum(128);
	ui.SpinBox_NumberCPUCore->setValue(1);
	ui.SpinBox_ClusteringDepth->setValue(4);
	ui.CheckBox_IrreversibleFit->setChecked(false);




	updateIrreversibleFit();

}

BrainModeling::~BrainModeling()
{
	delete pDynamicPETData;
	delete pMask;
	delete pAIF;

	delete[] DPETFileName;	
}

void BrainModeling::getInputPETDataPath()
{
	QFileDialog qfopen;
	QString fileName = qfopen.getOpenFileName(this,tr("Open Image"), SearchPath, tr("Dynamic PET (*.nii *.ots)"));	
	
	if ( fileName.isNull() == false )
	{
		QString ftype=fileName.right(3);


		if(ftype.compare("nii")==0||ftype.compare("ots")==0){

			if(pDynamicPETData!=NULL){
				delete pDynamicPETData;
			}

			ui.TextEdit_InputFileName->setText(fileName);

			strcpy(DPETFileName,fileName.toLatin1());

			pDynamicPETData=new ShiMatrix;

			bool bSuccess=false;

			if(ftype.compare("nii")==0){
				if(pDynamicPETData->ReadNiftiFile(DPETFileName)) bSuccess=true;
			}else{
				if(pDynamicPETData->ReadOTSFile(DPETFileName)) bSuccess=true;
				bSaveOTS=true;
			}
			

			if(bSuccess){
				//! Imran: !//
				if(pointerData!=NULL){
					delete pointerData;
				}
	
				pointerData = pDynamicPETData->dims;
				/*To be fixed:
				ui.ScrollBar_Slice->setRange(1,*(pointerData+2));
				ui.ScrollBar_Slice->update();
				ui.ScrollBar_Time->setRange(1,*(pointerData+3));
				ui.ScrollBar_Time->update();
				*/

				fileName.chop(4);
				ui.TextEdit_SavePrefix->setText(fileName);		

				SearchPath=QFileInfo(fileName).path();
			}else{
				QErrorMessage errorMessage;
				errorMessage.showMessage("File format not support!");
				errorMessage.exec();	
			}
		}else{
			QErrorMessage errorMessage;
			errorMessage.showMessage("File format not support!");
			errorMessage.exec();		
		}

	}

}

void BrainModeling::displayImageHei(QImage image)
{


			scene.addPixmap((QPixmap::fromImage(image)));
			scene.update();
			ui.GraphicsView_Display->setScene(&scene);
			ui.GraphicsView_Display->setBackgroundBrush(QBrush(Qt::lightGray));
			ui.GraphicsView_Display->show();


	//!Heirarchial Clustering code goes here!!!

	
}


void BrainModeling::z_axis(int z)
{
		int M = *(pointerData+0);
		int N = *(pointerData+1);
		int t = ui.ScrollBar_Time->value();
		
		if(imageMat!=NULL){
			delete imageMat;
		}
		imageMat = new ShiMatrix(M,N);

		for ( int m=0; m<M; m++ )
		{
			for ( int n=0; n<N; n++)
			{
				imageMat->set((pDynamicPETData->get(m,n,z,t)), m, n);
			}
		}

	 displayImageHei(RenderQImage(M,  N));
}

void BrainModeling::t_axis(int t)
{
	 	int M = *(pointerData+0);
		int N = *(pointerData+1);
		int z = ui.ScrollBar_Slice->value();
		if(imageMat!=NULL){
					delete imageMat;
				}
		imageMat = new ShiMatrix(M,N);
		
		for ( int m=0; m<M; m++ )
		{
			for ( int n=0; n<N; n++)
			{
				imageMat->set((pDynamicPETData->get(m,n,z,t)), m, n);
			}
		}
	 displayImageHei(RenderQImage(M,  N));
}

//! Convert a double in [0,1] to a qRbg value!//
uint BrainModeling::doubleToJetMap(float x)
{
    double  r = 255*((x >= 0.375 & x < 0.625) * (4.0 * x - 1.5) + (x >= 0.625 & x < 0.875) + (x >= 0.875) * (-4.0 * x + 4.5));
    double g = 255*((x >= 0.125 & x < 0.375) * (4.0 * x - 0.5) + (x >= 0.375 & x < 0.625) + (x >= 0.625 & x < 0.875) * (-4.0 * x + 3.5));
    double b = 255*((x < 0.125) * (4.0 * x + 0.5) + (x >= 0.125 & x < 0.375) + (x >= 0.375 & x < 0.625) * (-4.0 * x + 2.5));

    return qRgb((int)r,(int)g,(int)b);
}

QImage BrainModeling::RenderQImage(int M, int N)
{
    QImage imageRendered(M, N, QImage::Format_RGB32);

	//! Normalize 2D Matrix !//
	float maxPix = imageMat->get(0,0);
	    for ( int in=0; in<N; in++ )
    {
        for ( int im=0; im<M; im++)
		{
			maxPix = std::max(maxPix,imageMat->get(in,im));
		}
    }

	//! Render Matrix and convert it to an Image !//
    for ( int in=0; in<N; in++ )
    {
        for ( int im=0; im<M; im++)
		{
			imageRendered.setPixel(in,im, doubleToJetMap((imageMat->get(in, im))/maxPix));
		}
    }
    return imageRendered;
}


void BrainModeling::getAIFPath()
{
//	QString fileName = QFileDialog::getOpenFileName(this,tr("Open AIF"), path, tr("OTS Data (*.ots)"));
	QFileDialog qfopen;
	QString fileName = qfopen.getOpenFileName(this,tr("Open Image"), SearchPath, tr("OTS Data (*.ots)"));


	if ( fileName.isNull() == false )
	{
		if(pAIF!=NULL){
			delete pAIF;	
		}

		char *str = new char[200];
		strcpy(str,fileName.toLatin1());
		pAIF=new ShiMatrix;
		pAIF->ReadOTSFile(str);
		ui.TextEdit_AIFName->setText(fileName);
		delete[] str;
		SearchPath=QFileInfo(fileName).path();
	}
}


void BrainModeling::onComputing()
{

	if(bExcuted){ 
		time_t ctime;
		time(&ctime);
		if(difftime(ctime,CalculationEndTime)<3) return;
	}


	ui.Button_Computing->setEnabled(false);

	int NX=pDynamicPETData->dims[0];
	int NY=pDynamicPETData->dims[1];
	int NZ=pDynamicPETData->dims[2];
	int NT=pDynamicPETData->dims[3];
	pMask=new ShiMatrix(NX,NY,NZ,1);

	ComputeMaskData();


	char *str = new char[200];
	QString prefix=ui.TextEdit_SavePrefix->text();

	QString filename;
	
	QMessageBox msgBox;

	bool bSmoothPET=ui.CheckBox_SmoothPET->isChecked();
	if(bSmoothPET){
		ShiPreprocessImage::Temporal4DDataSmooth(pDynamicPETData,pAIF,pDynamicPETData,0.3,5);

		

		if(bSaveOTS){
			QString filename=prefix+QString("_Mask.ots");	
			strcpy(str,filename.toLatin1());
			pMask->WriteOTSFile(str);
			filename=prefix+QString("S.ots");	
			strcpy(str,filename.toLatin1());
			pDynamicPETData->WriteOTSFile(str);
		}else{
			QString filename=prefix+QString("_Mask.nii");	
			strcpy(str,filename.toLatin1());
			pMask->WriteNiftiFile(str,DPETFileName);
			filename=prefix+QString("S.nii");	
			strcpy(str,filename.toLatin1());
			pDynamicPETData->WriteNiftiFile(str,DPETFileName);
		}
		
//		msgBox.setText(QString("Smooth Finished!"));
//		msgBox.exec();	

		delete[] str;
		return;
	}
	


	double ub[5];
	double lb[5];
	double p[5];

	

	if(pDynamicPETData==NULL || pMask==NULL || pAIF==NULL) {
		QErrorMessage errorMessage;
		errorMessage.showMessage("Input Data is not Complete");
		errorMessage.exec();
		return;
	}


	QString tmp=ui.TextEdit_K1Init->text();
	p[0]=tmp.toDouble();
	tmp=ui.TextEdit_K2Init->text();
	p[1]=tmp.toDouble();
	tmp=ui.TextEdit_K3Init->text();
	p[2]=tmp.toDouble();
	tmp=ui.TextEdit_K4Init->text();
	p[3]=tmp.toDouble();
	tmp=ui.TextEdit_VBInit->text();
	p[4]=tmp.toDouble();


	tmp=ui.TextEdit_K1Min->text();
	lb[0]=tmp.toDouble();
	tmp=ui.TextEdit_K2Min->text();
	lb[1]=tmp.toDouble();
	tmp=ui.TextEdit_K3Min->text();
	lb[2]=tmp.toDouble();
	tmp=ui.TextEdit_K4Min->text();
	lb[3]=tmp.toDouble();
	tmp=ui.TextEdit_VBMin->text();
	lb[4]=tmp.toDouble();


	tmp=ui.TextEdit_K1Max->text();
	ub[0]=tmp.toDouble();
	tmp=ui.TextEdit_K2Max->text();
	ub[1]=tmp.toDouble();
	tmp=ui.TextEdit_K3Max->text();
	ub[2]=tmp.toDouble();
	tmp=ui.TextEdit_K4Max->text();
	ub[3]=tmp.toDouble();
	tmp=ui.TextEdit_VBMax->text();
	ub[4]=tmp.toDouble();

	int nCPUParallelCore=ui.SpinBox_NumberCPUCore->value();
	
	bool bWeightedFitting=ui.CheckBox_WeightedFit->isChecked();
	bool bClusterAvailable=ui.CheckBox_Segmented->isChecked();
	bool bIrreversibleFit=ui.CheckBox_IrreversibleFit->isChecked();
	int nSuperSample=ui.SpinBox_NumberSupersample->value();
	int NCluster=ui.SpinBox_NumberCluster->value();
	int NDepth=ui.SpinBox_ClusteringDepth->value();

	omp_set_num_threads(nCPUParallelCore);

	pResultK1=new ShiMatrix(NX,NY,NZ,1);
	pResultK2=new ShiMatrix(NX,NY,NZ,1);
	pResultK3=new ShiMatrix(NX,NY,NZ,1);
	pResultK4=new ShiMatrix(NX,NY,NZ,1);
	pResultVB=new ShiMatrix(NX,NY,NZ,1);
	pChiSquare=new ShiMatrix(NX,NY,NZ,1);






	time_t start,end;
	time(&start);
	

	ParallelModelingConsole pmconsole;
	pmconsole.Initialize(nSuperSample, NCluster, NDepth, bWeightedFitting, bIrreversibleFit, p,lb,ub);


	if(bClusterAvailable){
		pLabelMap=new ShiMatrix;
		pParaList=new ShiMatrix;

		if(bSaveOTS){
			filename=prefix+QString("_LabelMap.ots");	
			strcpy(str,filename.toLatin1());
			pLabelMap->ReadOTSFile(str);
		}else{
			filename=prefix+QString("_LabelMap.nii");	
			strcpy(str,filename.toLatin1());
			pLabelMap->ReadOTSFile(str);
			pLabelMap->ReadNiftiFile(str);
		}

		filename=prefix+QString("_ParaList.ots");	
		strcpy(str,filename.toLatin1());
		pParaList->ReadOTSFile(str);
		
		pmconsole.HierarchicalModelingWithClusteringInfo(pDynamicPETData,pMask,pAIF,pLabelMap,pParaList,pResultK1,pResultK2,pResultK3,pResultK4,pResultVB,pChiSquare);

	}else{


		pLabelMap=new ShiMatrix;
		pParaList=new ShiMatrix;

		pmconsole.HierarchicalClustering(pDynamicPETData,pMask,pAIF,pLabelMap,pParaList);

		if(bSaveOTS){
			filename=prefix+QString("_LabelMap.ots");	
			strcpy(str,filename.toLatin1());
			pLabelMap->WriteOTSFile(str);
		}else{
			filename=prefix+QString("_LabelMap.nii");	
			strcpy(str,filename.toLatin1());
			pLabelMap->WriteNiftiFile(str,DPETFileName);
		}


		filename=prefix+QString("_ParaList.ots");	
		strcpy(str,filename.toLatin1());
		pParaList->WriteOTSFile(str);


//		msgBox.setText("ParaList Saved!");
//		msgBox.exec();

		pmconsole.HierarchicalModelingWithClusteringInfo(pDynamicPETData,pMask,pAIF,pLabelMap,pParaList,pResultK1,pResultK2,pResultK3,pResultK4,pResultVB,pChiSquare);

	}

	
	if(bSaveOTS){
		filename=prefix+QString("_K1.ots");	
		strcpy(str,filename.toLatin1());
		pResultK1->WriteOTSFile(str);

		filename=prefix+QString("_K2.ots");	
		strcpy(str,filename.toLatin1());
		pResultK2->WriteOTSFile(str);

		filename=prefix+QString("_K3.ots");	
		strcpy(str,filename.toLatin1());
		pResultK3->WriteOTSFile(str);

		filename=prefix+QString("_K4.ots");	
		strcpy(str,filename.toLatin1());
		pResultK4->WriteOTSFile(str);

		filename=prefix+QString("_VB.ots");	
		strcpy(str,filename.toLatin1());
		pResultVB->WriteOTSFile(str);

		filename=prefix+QString("_ChiSquare.ots");	
		strcpy(str,filename.toLatin1());
		pChiSquare->WriteOTSFile(str);

	}else{
		filename=prefix+QString("_K1.nii");	
		strcpy(str,filename.toLatin1());
		pResultK1->WriteNiftiFile(str,DPETFileName);

		filename=prefix+QString("_K2.nii");	
		strcpy(str,filename.toLatin1());
		pResultK2->WriteNiftiFile(str,DPETFileName);

		filename=prefix+QString("_K3.nii");	
		strcpy(str,filename.toLatin1());
		pResultK3->WriteNiftiFile(str,DPETFileName);

		filename=prefix+QString("_K4.nii");	
		strcpy(str,filename.toLatin1());
		pResultK4->WriteNiftiFile(str,DPETFileName);

		filename=prefix+QString("_VB.nii");	
		strcpy(str,filename.toLatin1());
		pResultVB->WriteNiftiFile(str,DPETFileName);

		filename=prefix+QString("_ChiSquare.nii");	
		strcpy(str,filename.toLatin1());
		pChiSquare->WriteNiftiFile(str,DPETFileName);
/**

		filename=prefix+QString("_K1.ots");	
		strcpy(str,filename.toLatin1());
		pResultK1->WriteOTSFile(str);

		filename=prefix+QString("_K2.ots");	
		strcpy(str,filename.toLatin1());
		pResultK2->WriteOTSFile(str);

		filename=prefix+QString("_K3.ots");	
		strcpy(str,filename.toLatin1());
		pResultK3->WriteOTSFile(str);

		filename=prefix+QString("_K4.ots");	
		strcpy(str,filename.toLatin1());
		pResultK4->WriteOTSFile(str);

		filename=prefix+QString("_VB.ots");	
		strcpy(str,filename.toLatin1());
		pResultVB->WriteOTSFile(str);

		filename=prefix+QString("_ChiSquare.ots");	
		strcpy(str,filename.toLatin1());
		pChiSquare->WriteOTSFile(str);
**/
	}
	

	delete[] str;

	delete pLabelMap;
	delete pParaList;

	delete pResultK1;
	delete pResultK2;
	delete pResultK3;
	delete pResultK4;
	delete pResultVB;
	delete pChiSquare;



	ui.Button_Computing->setEnabled(true);
	time(&CalculationEndTime);
	bExcuted=true;

	time(&end);

	double comptime=difftime(end,start);

	
	msgBox.setText(QString("Computation Finished! Computation time: ")+QString::number(comptime)+QString(" sec."));
	msgBox.exec();

}


void BrainModeling::enableComputing()
{
	bExcuted=false;
	ui.ProgressBar_Computing->setValue(0);
}


void BrainModeling::updateIrreversibleFit()
{
	if(ui.CheckBox_IrreversibleFit->isChecked()){
		ui.TextEdit_K4Init->setEnabled(false);
		ui.TextEdit_K4Min->setEnabled(false);
		ui.TextEdit_K4Max->setEnabled(false);
	}else{
		ui.TextEdit_K4Init->setEnabled(true);
		ui.TextEdit_K4Min->setEnabled(true);
		ui.TextEdit_K4Max->setEnabled(true);

	}
}

void BrainModeling::ComputeMaskData()
{
	int NX=pDynamicPETData->dims[0];
	int NY=pDynamicPETData->dims[1];
	int NZ=pDynamicPETData->dims[2];
	int NT=pDynamicPETData->dims[3];

	
	double maxV=0;

	double vl;
	for(int ix=0; ix<NX; ix++){
		for(int iy=0; iy<NY; iy++){
			for(int iz=0; iz<NZ; iz++){
				vl=0;
				for(int it=0; it<NT; it++){
					vl+=pDynamicPETData->get(ix,iy,iz,it);
				}				
				pMask->set(vl,ix,iy,iz,0);
				if (vl>maxV) maxV=vl;
			}
		}
	}

	double Thresh=0.12*maxV;
	int nv=0;
	double meanValue=0;
	for(int ix=0; ix<NX; ix++){
		for(int iy=0; iy<NY; iy++){
			for(int iz=0; iz<NZ; iz++){
				vl=pMask->get(ix,iy,iz,0);
				if(vl>Thresh){ 
					meanValue+=vl;	
					nv++;
				}
			}
		}
	}
	meanValue/=nv;

	Thresh=0.12*meanValue;
	nv=0;
	meanValue=0;
	for(int ix=0; ix<NX; ix++){
		for(int iy=0; iy<NY; iy++){
			for(int iz=0; iz<NZ; iz++){
				vl=pMask->get(ix,iy,iz,0);
				if(vl>Thresh){ 
					meanValue+=vl;	
					nv++;
				}
			}
		}
	}
	meanValue/=nv;


	Thresh=0.2*meanValue;
	for(int ix=0; ix<NX; ix++){
		for(int iy=0; iy<NY; iy++){
			for(int iz=0; iz<NZ; iz++){
				vl=pMask->get(ix,iy,iz,0);
				if(vl>Thresh){ 
					pMask->set(1,ix,iy,iz,0);
				}else{
					pMask->set(0,ix,iy,iz,0);
				}
			}
		}
	}

}

/**
connect(ui.TextEdit_K1Init, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K2Init, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K3Init, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K4Init, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_VBInit, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K1Min, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K2Min, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K3Min, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K4Min, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_VBMin, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K1Max, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K2Max, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K3Max, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_K4Max, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_VBMax, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.SpinBox_NumberCluster, SIGNAL(valueChanged()),this, SLOT(enableComputing()));
connect(ui.SpinBox_NumberSupersample, SIGNAL(valueChanged()),this, SLOT(enableComputing()));
connect(ui.SpinBox_NumberCluster, SIGNAL(valueChanged()),this, SLOT(enableComputing()));
connect(ui.CheckBox_WeightedFit, SIGNAL(stateChanged()),this, SLOT(enableComputing()));
connect(ui.CheckBox_SmoothPET, SIGNAL(stateChanged()),this, SLOT(enableComputing()));
connect(ui.CheckBox_Segmented, SIGNAL(stateChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_InputFileName, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_MaskName, SIGNAL( textChanged()),this, SLOT(enableComputing()));
connect(ui.TextEdit_AIFName, SIGNAL( textChanged()),this, SLOT(enableComputing()));
**/
/**

void BrainModeling::getMaskPath()
{
//	QString fileName = QFileDialog::getOpenFileName(this,tr("Open Image"), NULL, tr("NIFTI Data (*.nii)"));
	QFileDialog qfopen;
	QString fileName = qfopen.getOpenFileName(this,tr("Open Image"), SearchPath, tr("Mask (*.nii *.ots)"));

	if ( fileName.isNull() == false )
	{
		QString ftype=fileName.right(3);
		if(ftype.compare("nii")==0||ftype.compare("ots")==0){

			if(pMask!=NULL){
				delete pMask;	
			}

			strcpy(DPETFileName,fileName.toLatin1());
		
			pMask=new ShiMatrix;

			bool bSuccess=false;

			if(ftype.compare("nii")==0){
				if(pMask->ReadNiftiFile(DPETFileName)) bSuccess=true;
			}else{
				if(pMask->ReadOTSFile(DPETFileName)) bSuccess=true;
			}

			if(bSuccess){
				ui.TextEdit_MaskName->setText(fileName);
				SearchPath=QFileInfo(fileName).path();
				
			}else{
				QErrorMessage errorMessage;
				errorMessage.showMessage("File format not support!");
				errorMessage.exec();	
			}
		}

	}

}
**/