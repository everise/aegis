// Type declarations for modules without @types packages

declare module 'remark-gfm' {
  import { Plugin } from 'unified';
  const remarkGfm: Plugin;
  export default remarkGfm;
}

// Radix UI modules
declare module '@radix-ui/react-avatar' {
  import * as React from 'react';
  
  interface AvatarProps extends React.ComponentPropsWithoutRef<'span'> {
    asChild?: boolean;
  }
  
  interface AvatarImageProps extends React.ComponentPropsWithoutRef<'img'> {
    asChild?: boolean;
    onLoadingStatusChange?: (status: 'idle' | 'loading' | 'loaded' | 'error') => void;
  }
  
  interface AvatarFallbackProps extends React.ComponentPropsWithoutRef<'span'> {
    asChild?: boolean;
    delayMs?: number;
  }
  
  const Root: React.ForwardRefExoticComponent<AvatarProps & React.RefAttributes<HTMLSpanElement>>;
  const Image: React.ForwardRefExoticComponent<AvatarImageProps & React.RefAttributes<HTMLImageElement>>;
  const Fallback: React.ForwardRefExoticComponent<AvatarFallbackProps & React.RefAttributes<HTMLSpanElement>>;
  
  export { Root, Image, Fallback };
}

declare module '@radix-ui/react-slot' {
  import * as React from 'react';
  
  interface SlotProps extends React.HTMLAttributes<HTMLElement> {
    children?: React.ReactNode;
  }
  
  const Slot: React.ForwardRefExoticComponent<SlotProps & React.RefAttributes<HTMLElement>>;
  export { Slot };
}

declare module '@radix-ui/react-select' {
  import * as React from 'react';
  
  interface SelectProps {
    children?: React.ReactNode;
    value?: string;
    defaultValue?: string;
    onValueChange?: (value: string) => void;
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
    dir?: 'ltr' | 'rtl';
    name?: string;
    disabled?: boolean;
    required?: boolean;
  }
  
  interface SelectTriggerProps extends React.ComponentPropsWithoutRef<'button'> {
    asChild?: boolean;
  }
  
  interface SelectContentProps extends React.ComponentPropsWithoutRef<'div'> {
    asChild?: boolean;
    position?: 'item-aligned' | 'popper';
    side?: 'top' | 'right' | 'bottom' | 'left';
    sideOffset?: number;
    align?: 'start' | 'center' | 'end';
    alignOffset?: number;
  }
  
  interface SelectItemProps extends React.ComponentPropsWithoutRef<'div'> {
    asChild?: boolean;
    value: string;
    disabled?: boolean;
    textValue?: string;
  }
  
  interface SelectValueProps extends React.ComponentPropsWithoutRef<'span'> {
    asChild?: boolean;
    placeholder?: React.ReactNode;
  }
  
  const Root: React.FC<SelectProps>;
  const Trigger: React.ForwardRefExoticComponent<SelectTriggerProps & React.RefAttributes<HTMLButtonElement>>;
  const Value: React.ForwardRefExoticComponent<SelectValueProps & React.RefAttributes<HTMLSpanElement>>;
  const Icon: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'span'> & { asChild?: boolean } & React.RefAttributes<HTMLSpanElement>>;
  const Portal: React.FC<{ children?: React.ReactNode; container?: HTMLElement }>;
  const Content: React.ForwardRefExoticComponent<SelectContentProps & React.RefAttributes<HTMLDivElement>>;
  const Viewport: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  const Group: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  const Label: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  const Item: React.ForwardRefExoticComponent<SelectItemProps & React.RefAttributes<HTMLDivElement>>;
  const ItemText: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'span'> & React.RefAttributes<HTMLSpanElement>>;
  const ItemIndicator: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'span'> & React.RefAttributes<HTMLSpanElement>>;
  const Separator: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  const ScrollUpButton: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  const ScrollDownButton: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'div'> & React.RefAttributes<HTMLDivElement>>;
  
  export { Root, Trigger, Value, Icon, Portal, Content, Viewport, Group, Label, Item, ItemText, ItemIndicator, Separator, ScrollUpButton, ScrollDownButton };
}

declare module '@radix-ui/react-tooltip' {
  import * as React from 'react';
  
  interface TooltipProviderProps {
    children?: React.ReactNode;
    delayDuration?: number;
    skipDelayDuration?: number;
    disableHoverableContent?: boolean;
  }
  
  interface TooltipProps {
    children?: React.ReactNode;
    open?: boolean;
    defaultOpen?: boolean;
    onOpenChange?: (open: boolean) => void;
    delayDuration?: number;
    disableHoverableContent?: boolean;
  }
  
  interface TooltipContentProps extends React.ComponentPropsWithoutRef<'div'> {
    asChild?: boolean;
    side?: 'top' | 'right' | 'bottom' | 'left';
    sideOffset?: number;
    align?: 'start' | 'center' | 'end';
    alignOffset?: number;
  }
  
  const Provider: React.FC<TooltipProviderProps>;
  const Root: React.FC<TooltipProps>;
  const Trigger: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'button'> & { asChild?: boolean } & React.RefAttributes<HTMLButtonElement>>;
  const Portal: React.FC<{ children?: React.ReactNode; container?: HTMLElement }>;
  const Content: React.ForwardRefExoticComponent<TooltipContentProps & React.RefAttributes<HTMLDivElement>>;
  const Arrow: React.ForwardRefExoticComponent<React.ComponentPropsWithoutRef<'svg'> & React.RefAttributes<SVGSVGElement>>;
  
  export { Provider, Root, Trigger, Portal, Content, Arrow };
}

// axios module
declare module 'axios' {
  export interface AxiosRequestConfig {
    url?: string;
    method?: string;
    baseURL?: string;
    headers?: Record<string, string>;
    params?: Record<string, unknown>;
    data?: unknown;
    timeout?: number;
    withCredentials?: boolean;
  }

  export interface InternalAxiosRequestConfig extends AxiosRequestConfig {
    headers: Record<string, string> & {
      Authorization?: string;
    };
  }

  export interface AxiosResponse<T = unknown> {
    data: T;
    status: number;
    statusText: string;
    headers: Record<string, string>;
    config: AxiosRequestConfig;
  }

  export interface AxiosError<T = unknown> extends Error {
    config?: AxiosRequestConfig;
    code?: string;
    request?: unknown;
    response?: AxiosResponse<T>;
    isAxiosError: boolean;
  }

  export interface AxiosInstance {
    defaults: AxiosRequestConfig;
    interceptors: {
      request: {
        use(
          onFulfilled?: (config: InternalAxiosRequestConfig) => InternalAxiosRequestConfig | Promise<InternalAxiosRequestConfig>,
          onRejected?: (error: unknown) => unknown
        ): number;
        eject(id: number): void;
      };
      response: {
        use(
          onFulfilled?: (response: AxiosResponse) => AxiosResponse | Promise<AxiosResponse>,
          onRejected?: (error: AxiosError) => unknown
        ): number;
        eject(id: number): void;
      };
    };
    get<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
    post<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
    put<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
    delete<T = unknown>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
    patch<T = unknown>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<AxiosResponse<T>>;
  }

  export function create(config?: AxiosRequestConfig): AxiosInstance;
  
  const axios: AxiosInstance & {
    create(config?: AxiosRequestConfig): AxiosInstance;
    isAxiosError(payload: unknown): payload is AxiosError;
  };
  
  export default axios;
}

// class-variance-authority module
declare module 'class-variance-authority' {
  type ClassValue = string | number | null | undefined | boolean | ClassValue[] | Record<string, unknown>;
  
  type ConfigSchema = Record<string, Record<string, ClassValue>>;
  
  type StringToBoolean<T> = T extends 'true' | 'false' ? boolean : T;
  
  type ConfigVariants<T extends ConfigSchema> = {
    [K in keyof T]?: StringToBoolean<keyof T[K]> | null | undefined;
  };
  
  type Props<T extends ConfigSchema> = ConfigVariants<T> & { className?: string };
  
  export function cva<T extends ConfigSchema>(
    base?: ClassValue,
    config?: {
      variants?: T;
      compoundVariants?: Array<ConfigVariants<T> & { className?: ClassValue }>;
      defaultVariants?: ConfigVariants<T>;
    }
  ): (props?: Props<T>) => string;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export type VariantProps<T extends (...args: any[]) => any> = Omit<
    NonNullable<Parameters<T>[0]>,
    'className'
  >;
}
