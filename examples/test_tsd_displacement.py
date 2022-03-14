# [THINSHELLDEMONS] Read Mesh and Create TSD object

# Build the Module with Python Wrapping and import the itk package which is available under Wrapping/Generators/Python
#import sys
#sys.path.append('/home/pranjal.sahu/ITKPR7/ITK/ITK-build/Wrapping/Generators/Python')

import itkConfig
itkConfig.LazyLoading = False

import numpy as np
import itk
import math

fixedMesh = itk.meshread('../Baseline/fixedMesh.vtk', itk.D)
movingMesh = itk.meshread('../Baseline/movingMesh.vtk', itk.D)

fixedMesh.BuildCellLinks()
movingMesh.BuildCellLinks()

PixelType = itk.D
Dimension = 3

MeshType = itk.Mesh[itk.D, Dimension]
FixedImageType = itk.Image[PixelType, Dimension]
MovingImageType = itk.Image[PixelType, Dimension]


# For getting the Bounding Box
ElementIdentifierType = itk.UL
CoordType = itk.F
Dimension = 3

VecContType = itk.VectorContainer[
    ElementIdentifierType, itk.Point[CoordType, Dimension]
]
bounding_box = itk.BoundingBox[ElementIdentifierType, Dimension, CoordType, VecContType].New()
bounding_box.SetPoints(movingMesh.GetPoints())
bounding_box.ComputeBoundingBox()

minBounds = np.array(bounding_box.GetMinimum())
maxBounds = np.array(bounding_box.GetMaximum())

imageDiagonal = 5
spacing = np.sqrt(bounding_box.GetDiagonalLength2()) / imageDiagonal
diff = maxBounds - minBounds

fixedImageSize = [0]*3
fixedImageSize[0] = math.ceil( 1.2 * diff[0] / spacing )
fixedImageSize[1] = math.ceil( 1.2 * diff[1] / spacing )
fixedImageSize[2] = math.ceil( 1.2 * diff[2] / spacing )

fixedImageOrigin = [0]*3
fixedImageOrigin[0] = minBounds[0] - 0.1 * diff[0]
fixedImageOrigin[1] = minBounds[1] - 0.1 * diff[1]
fixedImageOrigin[2] = minBounds[2] - 0.1 * diff[2]

fixedImageSpacing = np.ones(3)*spacing
fixedImageDirection = np.identity(3)


fixedImage = FixedImageType.New()
fixedImage.SetRegions(fixedImageSize)
fixedImage.SetOrigin( fixedImageOrigin )
fixedImage.SetDirection( fixedImageDirection )
fixedImage.SetSpacing( fixedImageSpacing )
fixedImage.Allocate()


# Displacement Transform Object
TransformType = itk.DisplacementFieldTransform[itk.D, Dimension]
transform = TransformType.New()
field = itk.Image[itk.Vector[itk.D, Dimension], Dimension].New()
field.SetRegions( fixedImageSize )
field.SetOrigin( fixedImageOrigin )
field.SetDirection( fixedImageDirection )
field.SetSpacing( fixedImageSpacing )
field.Allocate()
transform.SetDisplacementField(field)

print('Transform Created')
print(transform)


MetricType = itk.ThinShellDemonsMetricv4.MD3
metric = MetricType.New()
metric.SetStretchWeight(1)
metric.SetBendWeight(5)
metric.SetGeometricFeatureWeight(10)
metric.UseConfidenceWeightingOn()
metric.UseMaximalDistanceConfidenceSigmaOn()
metric.UpdateFeatureMatchingAtEachIterationOff()
metric.SetMovingTransform(transform)
# Reversed due to using points instead of an image
# to keep semantics the same as in itkThinShellDemonsTest.cxx
# For the ThinShellDemonsMetricv4 the fixed mesh is regularized
metric.SetFixedPointSet(movingMesh)
metric.SetMovingPointSet(fixedMesh)
metric.SetVirtualDomainFromImage(fixedImage)
metric.Initialize()

print('TSD Metric Created')

shiftScaleEstimator = itk.RegistrationParameterScalesFromPhysicalShift[MetricType].New()
shiftScaleEstimator.SetMetric(metric)
shiftScaleEstimator.SetVirtualDomainPointSet(metric.GetVirtualTransformedPointSet())


optimizer = itk.ConjugateGradientLineSearchOptimizerv4Template.D.New()
optimizer.SetNumberOfIterations( 50 )
optimizer.SetScalesEstimator( shiftScaleEstimator )
optimizer.SetMaximumStepSizeInPhysicalUnits( 0.5 )
optimizer.SetMinimumConvergenceValue( 0.0 )
optimizer.SetConvergenceWindowSize( 10 )

def iteration_update():
    metric_value = optimizer.GetValue()
    current_parameters = optimizer.GetCurrentPosition()
    print(f"Metric: {metric_value:.8g}")

iteration_command = itk.PyCommand.New()
iteration_command.SetCommandCallable(iteration_update)
optimizer.AddObserver(itk.IterationEvent(), iteration_command)

print('Optimizer created')


AffineRegistrationType = itk.ImageRegistrationMethodv4.REGv4D3D3TD3D3MD3.New()
registration = AffineRegistrationType.New()
registration.SetNumberOfLevels(1)
registration.SetObjectName("registration")
registration.SetFixedPointSet(movingMesh)
registration.SetMovingPointSet(fixedMesh)
registration.SetInitialTransform(transform)
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)

print('Registration Object created')
print('Initial Value of Metric ', metric.GetValue())

try:
    registration.Update()
except e:
    print('Error is ', e)

print('Final Value of Metric ', metric.GetValue())

finalTransform = registration.GetModifiableTransform()
numberOfPoints = movingMesh.GetNumberOfPoints()
for n in range(0, numberOfPoints):
    movingMesh.SetPoint(n, finalTransform.TransformPoint(movingMesh.GetPoint(n)))

itk.meshwrite(movingMesh, "displacedMovingMesh.vtk")
