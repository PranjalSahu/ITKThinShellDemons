/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include <cstdlib>

#include "itkCommand.h"
#include "itkThinShellDemonsMetricv4.h"
#include "itkImageRegistrationMethodv4.h"
#include "itkBSplineTransform.h"
#include "itkBSplineTransformInitializer.h"
//#include "itkMeshDisplacementTransform.h"
#include "itkConjugateGradientLineSearchOptimizerv4.h"
#include "itkRegularStepGradientDescentOptimizerv4.h"
#include "itkGradientDescentOptimizerv4.h"
#include "itkLBFGS2Optimizerv4.h"
#include "itkMesh.h"
#include "itkMeshFileReader.h"
#include "itkMeshFileWriter.h"

namespace{
template<typename TFilter>
class CommandIterationUpdate : public itk::Command
{
public:
  typedef  CommandIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>   Pointer;
  itkNewMacro( Self );
protected:
  CommandIterationUpdate() {};
public:
  void Execute(itk::Object *caller, const itk::EventObject & event) override
    {
    Execute( (const itk::Object *) caller, event);
    }

  void Execute(const itk::Object * object, const itk::EventObject & event) override
    {
    if( typeid( event ) != typeid( itk::IterationEvent ) )
      {
      return;
      }
    const auto * optimizer = dynamic_cast< const TFilter * >( object );

    if( !optimizer )
      {
        itkGenericExceptionMacro( "Error dynamic_cast failed" );
      }
    
    std::cout << "It: " << optimizer->GetCurrentIteration() << std::endl;
    std::cout << "Metric1 value: " << optimizer->GetCurrentMetricValue() << std::endl;
    std::cout << "Parameters : " << optimizer->GetCurrentPosition()  << std::endl;
    }
};
}

int itkThinShellDemonsTestv4_BSpline( int args, char *argv [])
{
  const unsigned int Dimension = 3;
  using MeshType = itk::Mesh<double, Dimension>;
  using PointsContainerPointer = MeshType::PointsContainerPointer;
  
  using ReaderType = itk::MeshFileReader<MeshType>;
  using WriterType = itk::MeshFileWriter<MeshType>;

  /*
  Initialize fixed mesh polydata reader
  */
  ReaderType::Pointer fixedPolyDataReader = ReaderType::New();
  fixedPolyDataReader->SetFileName("/home/pranjal.sahu/SlicerMorph/fixedMesh.vtk");
  try
  {
    fixedPolyDataReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Error during Fixed Mesh Update() " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  MeshType::Pointer fixedMesh = fixedPolyDataReader->GetOutput();

  /*
  Initialize moving mesh polydata reader
  */
  ReaderType::Pointer  movingPolyDataReader = ReaderType::New();
  movingPolyDataReader->SetFileName("/home/pranjal.sahu/SlicerMorph/ransac_result2.vtk");
  try
  {
    movingPolyDataReader->Update();
  }
  catch( itk::ExceptionObject & excp )
  {
    std::cerr << "Error during Moving Mesh Update() " << std::endl;
    std::cerr << excp << std::endl;
    return EXIT_FAILURE;
  }
  MeshType::Pointer movingMesh = movingPolyDataReader->GetOutput();

  /* Building the Cell Links to compute the neighbours later */
  fixedMesh->BuildCellLinks();
  movingMesh->BuildCellLinks();

  using PixelType = double;
  using FixedImageType = itk::Image<PixelType, Dimension>;
  using MovingImageType = itk::Image<PixelType, Dimension>;


  FixedImageType::SizeType fixedImageSize;
  FixedImageType::PointType fixedImageOrigin;
  FixedImageType::DirectionType fixedImageDirection;
  FixedImageType::SpacingType fixedImageSpacing;

  using PointIdentifier = MeshType::PointIdentifier;
  using BoundingBoxType = itk::BoundingBox<PointIdentifier, Dimension>;
  BoundingBoxType::Pointer boundingBox = BoundingBoxType::New();
  PointsContainerPointer points = movingMesh->GetPoints();
  boundingBox->SetPoints(points);
  boundingBox->ComputeBoundingBox();
  typename BoundingBoxType::PointType minBounds = boundingBox->GetMinimum();
  typename BoundingBoxType::PointType maxBounds = boundingBox->GetMaximum();

  int imageDiagonal = 200;
  double spacing = sqrt(boundingBox->GetDiagonalLength2()) / imageDiagonal;
  auto diff = maxBounds - minBounds;
  fixedImageSize[0] = ceil( 1.2 * diff[0] / spacing );
  fixedImageSize[1] = ceil( 1.2 * diff[1] / spacing );
  fixedImageSize[2] = ceil( 1.2 * diff[2] / spacing );
  fixedImageOrigin[0] = minBounds[0] - 0.1*diff[0];
  fixedImageOrigin[1] = minBounds[1] - 0.1*diff[1];
  fixedImageOrigin[2] = minBounds[2] - 0.1*diff[2];
  fixedImageDirection.SetIdentity();
  fixedImageSpacing.Fill( spacing );

  FixedImageType::Pointer fixedImage = FixedImageType::New();
  fixedImage->SetRegions( fixedImageSize );
  fixedImage->SetOrigin( fixedImageOrigin );
  fixedImage->SetDirection( fixedImageDirection );
  fixedImage->SetSpacing( fixedImageSpacing );
  fixedImage->Allocate();


  // using TransformType = itk::DisplacementFieldTransform<double, Dimension>;
  // auto transform = TransformType::New();
  // using  DisplacementFieldType = TransformType::DisplacementFieldType;
  // DisplacementFieldType::Pointer field = DisplacementFieldType::New();
  // field->SetRegions( fixedImageSize );
  // field->SetOrigin( fixedImageOrigin );
  // field->SetDirection( fixedImageDirection );
  // field->SetSpacing( fixedImageSpacing );
  // field->Allocate();
  // transform->SetDisplacementField(field);


  const unsigned int     SpaceDimension = 3;
  constexpr unsigned int SplineOrder = 3;
  using CoordinateRepType = double;

  using TransformType = itk::BSplineTransform<CoordinateRepType, SpaceDimension, SplineOrder>;

  auto transform = TransformType::New();

  using InitializerType = itk::BSplineTransformInitializer<TransformType, FixedImageType>;
  auto transformInitializer = InitializerType::New();


  int numberOfGridNodesInOneDimension = 6;

  TransformType::MeshSizeType meshSize;
  meshSize.Fill(numberOfGridNodesInOneDimension - SplineOrder);

  //auto transformInitializer = InitializerType::New();
  transformInitializer->SetTransform(transform);
  transformInitializer->SetImage(fixedImage);
  transformInitializer->SetTransformDomainMeshSize(meshSize);
  transformInitializer->InitializeTransform();


  std::cout << "Number of Transform Parameters " << transform->GetNumberOfParameters() << std::endl;
  std::cout << "Number of fixed points " << fixedMesh->GetNumberOfPoints() << std::endl;
  std::cout << "Number of moving points " << movingMesh->GetNumberOfPoints() << std::endl;

  using PointSetMetricType = itk::ThinShellDemonsMetricv4<MeshType, MeshType> ;
  PointSetMetricType::Pointer metric = PointSetMetricType::New();
  metric->SetStretchWeight(0.0001);
  metric->SetBendWeight(0.0001);
  metric->SetGeometricFeatureWeight(0.0001);
  metric->UseConfidenceWeightingOn();
  metric->UseMaximalDistanceConfidenceSigmaOff();
  metric->UpdateFeatureMatchingAtEachIterationOn();
  metric->SetMovingTransform( transform );
  //Reversed due to using points instead of an image
  //to keep semantics the same as in itkThinShellDemonsTest.cxx
  //For the ThinShellDemonsMetricv4 the fixed mesh is
  //regularized
  metric->SetFixedPointSet( movingMesh );
  metric->SetMovingPointSet( fixedMesh );
  metric->SetVirtualDomainFromImage( fixedImage );
  metric->Initialize();

  // Scales estimator
  using ScalesType = itk::RegistrationParameterScalesFromPhysicalShift< PointSetMetricType >;
  ScalesType::Pointer shiftScaleEstimator = ScalesType::New();
  shiftScaleEstimator->SetMetric( metric );
  // Needed with pointset metrics
  shiftScaleEstimator->SetVirtualDomainPointSet( metric->GetVirtualTransformedPointSet() );


  // optimizer

  // Does currently not support local transform
  // but change requested in:
  // https://github.com/InsightSoftwareConsortium/ITK/pull/2372
/*
  typedef itk::LBFGS2Optimizerv4 OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetScalesEstimator( shiftScaleEstimator );
*/

  // typedef itk::ConjugateGradientLineSearchOptimizerv4 OptimizerType;
  // OptimizerType::Pointer optimizer = OptimizerType::New();
  // optimizer->SetNumberOfIterations( 50 );
  // optimizer->SetScalesEstimator( shiftScaleEstimator );
  // optimizer->SetMaximumStepSizeInPhysicalUnits( 0.5 );
  // optimizer->SetMinimumConvergenceValue( 0.0 );
  // optimizer->SetConvergenceWindowSize( 10 );
  // optimizer->SetMetric(metric);


  //typedef itk::GradientDescentOptimizerv4 OptimizerType;
  typedef itk::RegularStepGradientDescentOptimizerv4<double> OptimizerType;
  OptimizerType::Pointer optimizer = OptimizerType::New();
  optimizer->SetNumberOfIterations( 25);
  //optimizer->SetScalesEstimator( shiftScaleEstimator );
  optimizer->SetMaximumStepSizeInPhysicalUnits(10);
  optimizer->SetLearningRate(10);
  //optimizer->SetMinimumConvergenceValue( 0.0 );
  //optimizer->SetConvergenceWindowSize( 10 );
  optimizer->SetMetric(metric);
  //optimizer-

  std::cout << "Optimizer Setting done " << std::endl;
  using CommandType = CommandIterationUpdate<OptimizerType>;
  CommandType::Pointer observer = CommandType::New();
  optimizer->AddObserver( itk::IterationEvent(), observer );

  using RegistrationType = itk::ImageRegistrationMethodv4<FixedImageType,
        MovingImageType, TransformType, FixedImageType, MeshType>;
  RegistrationType::Pointer registration = RegistrationType::New();
  registration->SetNumberOfLevels(1);
  registration->SetObjectName("registration");
  registration->SetFixedPointSet(movingMesh);
  registration->SetMovingPointSet(fixedMesh);
  registration->SetInitialTransform(transform);
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);

  std::cout << "Start Value = " << metric->GetValue() << std::endl;
  try
    {
      //optimizer->StartOptimization();
      registration->Update();
    }
  catch( itk::ExceptionObject &e )
    {
    std::cerr << "Exception caught: " << e << std::endl;
    return EXIT_FAILURE;
    }

  TransformType::Pointer tx = registration->GetModifiableTransform();
  metric->SetTransform(tx);
  std::cout << "Solution Value= " << metric->GetValue() << std::endl;
  for (PointIdentifier n = 0; n < movingMesh->GetNumberOfPoints(); n++)
  {
    movingMesh->SetPoint(n, tx->TransformPoint(movingMesh->GetPoint(n)));
  }

  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(movingMesh);
  writer->SetFileName("bsplineMovingMesh2.vtk");
  writer->Update();
  return EXIT_SUCCESS;
}
